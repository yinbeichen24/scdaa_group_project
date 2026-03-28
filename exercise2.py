import os
import math
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# 0. Reproducibility
# ============================================================

def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. LQR benchmark solver from Exercise 1.1
# ============================================================

class LQRProblem:
    """
    Solve the 2D LQR benchmark problem:

        dX_s = (H X_s + M a_s) ds + sigma dW_s

    with cost

        J^a(t,x) = E [ integral_t^T (X_s^T C X_s + a_s^T D a_s) ds + X_T^T R X_T ]

    The value function is
        v(t,x) = x^T S(t) x + integral_t^T tr(sigma sigma^T S(r)) dr

    where S solves the Riccati ODE
        S'(t) = -2 H^T S + S M D^{-1} M^T S - C
        S(T) = R

    and the optimal Markov control is
        a(t,x) = -D^{-1} M^T S(t) x
    """

    def __init__(self, H, M, sigma, C, D, R, T, device="cpu", dtype=torch.float32):
        self.H = np.asarray(H, dtype=float)
        self.M = np.asarray(M, dtype=float)
        self.sigma = np.asarray(sigma, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.R = np.asarray(R, dtype=float)
        self.T = float(T)

        self.D_inv = np.linalg.inv(self.D)

        self.device = device
        self.dtype = dtype

        self.time_grid = None
        self.S_grid = None
        self.g_grid = None

    def _riccati_rhs(self, t, y):
        S = y.reshape(2, 2)
        rhs = -2.0 * self.H.T @ S + S @ self.M @ self.D_inv @ self.M.T @ S - self.C
        return rhs.reshape(-1)

    def solve_riccati(self, time_grid):
        """
        Solve Riccati ODE backward in time on an increasing grid ending at T.
        Also compute
            g(t) = integral_t^T tr(sigma sigma^T S(r)) dr
        by trapezoidal rule.
        """
        time_grid = np.asarray(time_grid, dtype=float)
        assert np.all(np.diff(time_grid) > 0), "time_grid must be strictly increasing"
        assert abs(time_grid[-1] - self.T) < 1e-12, "time_grid must end at T"

        yT = self.R.reshape(-1)

        sol = solve_ivp(
            fun=self._riccati_rhs,
            t_span=(self.T, time_grid[0]),
            y0=yT,
            t_eval=time_grid[::-1],
            rtol=1e-9,
            atol=1e-11,
            method="RK45"
        )

        if not sol.success:
            raise RuntimeError("Riccati ODE solver failed.")

        S_rev = sol.y.T[::-1]
        S_grid = S_rev.reshape(-1, 2, 2)

        # numerical symmetrisation
        S_grid = 0.5 * (S_grid + np.transpose(S_grid, (0, 2, 1)))

        A = self.sigma @ self.sigma.T
        tr_vals = np.array([np.trace(A @ S) for S in S_grid])

        g_grid = np.zeros_like(tr_vals)
        for i in range(len(time_grid) - 2, -1, -1):
            dt = time_grid[i + 1] - time_grid[i]
            g_grid[i] = g_grid[i + 1] + 0.5 * dt * (tr_vals[i] + tr_vals[i + 1])

        self.time_grid = time_grid
        self.S_grid = S_grid
        self.g_grid = g_grid

    def _interp_S_and_g(self, t_values):
        t_values = np.asarray(t_values, dtype=float)

        S_out = np.zeros((len(t_values), 2, 2))
        g_out = np.zeros(len(t_values))

        for i, t in enumerate(t_values):
            if t <= self.time_grid[0]:
                S_out[i] = self.S_grid[0]
                g_out[i] = self.g_grid[0]
            elif t >= self.time_grid[-1]:
                S_out[i] = self.S_grid[-1]
                g_out[i] = self.g_grid[-1]
            else:
                j = np.searchsorted(self.time_grid, t) - 1
                t0, t1 = self.time_grid[j], self.time_grid[j + 1]
                w = (t - t0) / (t1 - t0)

                S_out[i] = (1.0 - w) * self.S_grid[j] + w * self.S_grid[j + 1]
                g_out[i] = (1.0 - w) * self.g_grid[j] + w * self.g_grid[j + 1]

        return S_out, g_out

    def value_function(self, t_batch, x_batch):
        """
        t_batch: torch tensor of shape (batch,)
        x_batch: torch tensor of shape (batch, 1, 2)

        returns:
            torch tensor of shape (batch, 1)
        """
        t_np = t_batch.detach().cpu().numpy()
        x_np = x_batch.detach().cpu().numpy().reshape(-1, 2)

        S_np, g_np = self._interp_S_and_g(t_np)
        vals = np.einsum("bi,bij,bj->b", x_np, S_np, x_np) + g_np
        vals = vals.reshape(-1, 1)

        return torch.tensor(vals, dtype=self.dtype, device=self.device)

    def markov_control(self, t_batch, x_batch):
        """
        t_batch: torch tensor of shape (batch,)
        x_batch: torch tensor of shape (batch, 1, 2)

        returns:
            torch tensor of shape (batch, 2)
        """
        t_np = t_batch.detach().cpu().numpy()
        x_np = x_batch.detach().cpu().numpy().reshape(-1, 2)

        S_np, _ = self._interp_S_and_g(t_np)
        controls = np.zeros((len(t_np), 2))

        for i in range(len(t_np)):
            controls[i] = - self.D_inv @ self.M.T @ S_np[i] @ x_np[i]

        return torch.tensor(controls, dtype=self.dtype, device=self.device)


# ============================================================
# 2. Networks for Exercise 2
# ============================================================

class DGMLayer(nn.Module):
    """
    One DGM-style hidden layer.

    Inputs:
        x: (batch, input_dim)     original input [t, x1, x2]
        s: (batch, hidden_dim)    hidden state

    Output:
        updated hidden state of shape (batch, hidden_dim)
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()

        self.U_z = nn.Linear(input_dim, hidden_dim)
        self.W_z = nn.Linear(hidden_dim, hidden_dim)

        self.U_g = nn.Linear(input_dim, hidden_dim)
        self.W_g = nn.Linear(hidden_dim, hidden_dim)

        self.U_r = nn.Linear(input_dim, hidden_dim)
        self.W_r = nn.Linear(hidden_dim, hidden_dim)

        self.U_h = nn.Linear(input_dim, hidden_dim)
        self.W_h = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, s):
        z = torch.tanh(self.U_z(x) + self.W_z(s))
        g = torch.tanh(self.U_g(x) + self.W_g(s))
        r = torch.tanh(self.U_r(x) + self.W_r(s))
        h = torch.tanh(self.U_h(x) + self.W_h(s * r))
        s_new = (1.0 - g) * h + z * s
        return s_new


class NetDGM(nn.Module):
    """
    DGM-style network for approximating the value function v(t,x).

    Input shape:
        (batch, 3) with columns [t, x1, x2]

    Output shape:
        (batch, 1)
    """
    def __init__(self, input_dim=3, hidden_dim=100, output_dim=1, n_layers=3):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh()
        )

        self.dgm_layers = nn.ModuleList(
            [DGMLayer(input_dim, hidden_dim) for _ in range(n_layers)]
        )

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        s = self.input_layer(x)
        for layer in self.dgm_layers:
            s = layer(x, s)
        out = self.output_layer(s)
        return out


class FFN(nn.Module):
    """
    Standard feedforward network for approximating the control a(t,x).

    Input shape:
        (batch, 3)

    Output shape:
        (batch, 2)
    """
    def __init__(self, layer_sizes, activation=nn.Tanh):
        super().__init__()
        layers = []

        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(activation())

        layers.append(nn.Linear(layer_sizes[-2], layer_sizes[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# 3. Data generation for supervised learning
# ============================================================

def generate_value_data(lqr, n_samples=20000, T=1.0, low=-3.0, high=3.0, seed=1234, device="cpu"):
    """
    Generate training data for Exercise 2.1.

    Inputs sampled uniformly:
        t ~ U([0,T])
        x ~ U([-3,3]^2)

    Returns:
        X_input: (N, 3) = [t, x1, x2]
        y:       (N, 1) = value function labels
    """
    rng = np.random.default_rng(seed)

    t = rng.uniform(0.0, T, size=n_samples).astype(np.float32)
    x = rng.uniform(low, high, size=(n_samples, 2)).astype(np.float32)

    t_torch = torch.tensor(t, dtype=torch.float32, device=device)
    x_torch = torch.tensor(x[:, None, :], dtype=torch.float32, device=device)

    y = lqr.value_function(t_torch, x_torch).detach().cpu()

    X_input = torch.tensor(np.column_stack([t, x]), dtype=torch.float32)

    return X_input, y


def generate_control_data(lqr, n_samples=20000, T=1.0, low=-3.0, high=3.0, seed=1234, device="cpu"):
    """
    Generate training data for Exercise 2.2.

    Inputs sampled uniformly:
        t ~ U([0,T])
        x ~ U([-3,3]^2)

    Returns:
        X_input: (N, 3) = [t, x1, x2]
        y:       (N, 2) = control labels
    """
    rng = np.random.default_rng(seed)

    t = rng.uniform(0.0, T, size=n_samples).astype(np.float32)
    x = rng.uniform(low, high, size=(n_samples, 2)).astype(np.float32)

    t_torch = torch.tensor(t, dtype=torch.float32, device=device)
    x_torch = torch.tensor(x[:, None, :], dtype=torch.float32, device=device)

    y = lqr.markov_control(t_torch, x_torch).detach().cpu()

    X_input = torch.tensor(np.column_stack([t, x]), dtype=torch.float32)

    return X_input, y


# ============================================================
# 4. Training utilities
# ============================================================

def train_supervised_model(
    model,
    X,
    y,
    batch_size=256,
    lr=1e-3,
    epochs=200,
    device="cpu",
    print_every=20
):
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)

        epoch_loss = running_loss / len(dataset)
        loss_history.append(epoch_loss)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:4d} | Loss = {epoch_loss:.6e}")

    return model, loss_history


def plot_loss(loss_history, title, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def evaluate_value_model(model, lqr, n_test=5, T=1.0, low=-3.0, high=3.0, seed=2026, device="cpu"):
    rng = np.random.default_rng(seed)

    t = rng.uniform(0.0, T, size=n_test).astype(np.float32)
    x = rng.uniform(low, high, size=(n_test, 2)).astype(np.float32)

    X_input = torch.tensor(np.column_stack([t, x]), dtype=torch.float32, device=device)

    t_torch = torch.tensor(t, dtype=torch.float32, device=device)
    x_torch = torch.tensor(x[:, None, :], dtype=torch.float32, device=device)
    y_true = lqr.value_function(t_torch, x_torch).detach().cpu().numpy().reshape(-1)

    model.eval()
    with torch.no_grad():
        y_pred = model(X_input).detach().cpu().numpy().reshape(-1)

    print("\nValue network quick check:")
    for i in range(n_test):
        print(f"Point {i+1}: t={t[i]:.3f}, x={x[i]}")
        print(f"   true value = {y_true[i]:.6f}")
        print(f"   pred value = {y_pred[i]:.6f}")
        print(f"   abs error  = {abs(y_true[i] - y_pred[i]):.6e}")


def evaluate_control_model(model, lqr, n_test=5, T=1.0, low=-3.0, high=3.0, seed=2027, device="cpu"):
    rng = np.random.default_rng(seed)

    t = rng.uniform(0.0, T, size=n_test).astype(np.float32)
    x = rng.uniform(low, high, size=(n_test, 2)).astype(np.float32)

    X_input = torch.tensor(np.column_stack([t, x]), dtype=torch.float32, device=device)

    t_torch = torch.tensor(t, dtype=torch.float32, device=device)
    x_torch = torch.tensor(x[:, None, :], dtype=torch.float32, device=device)
    y_true = lqr.markov_control(t_torch, x_torch).detach().cpu().numpy()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_input).detach().cpu().numpy()

    print("\nControl network quick check:")
    for i in range(n_test):
        print(f"Point {i+1}: t={t[i]:.3f}, x={x[i]}")
        print(f"   true control = {y_true[i]}")
        print(f"   pred control = {y_pred[i]}")
        print(f"   L2 error     = {np.linalg.norm(y_true[i] - y_pred[i]):.6e}")


# ============================================================
# 5. Main script
# ============================================================

def main():
    set_seed(1234)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --------------------------------------------------------
    # Replace these matrices by the Example 4.13 matrices
    # from your lecture notes if needed.
    # --------------------------------------------------------
    T = 1.0

    H = np.array([
        [0.10, 0.00],
        [0.00, 0.20]
    ])

    M = np.array([
        [1.00, 0.00],
        [0.00, 1.00]
    ])

    sigma = np.array([
        [0.30, 0.00],
        [0.00, 0.20]
    ])

    C = np.array([
        [1.00, 0.00],
        [0.00, 1.00]
    ])

    D = np.array([
        [1.00, 0.00],
        [0.00, 1.00]
    ])

    R = np.array([
        [1.00, 0.00],
        [0.00, 1.00]
    ])

    # --------------------------------------------------------
    # Solve benchmark LQR problem (Exercise 1.1 output)
    # --------------------------------------------------------
    lqr = LQRProblem(H, M, sigma, C, D, R, T, device=device)
    riccati_grid = np.linspace(0.0, T, 2001)
    lqr.solve_riccati(riccati_grid)
    print("Solved Riccati ODE.")

    os.makedirs("outputs", exist_ok=True)

    # --------------------------------------------------------
    # Exercise 2.1: supervised learning of value function
    # --------------------------------------------------------
    print("\n=== Exercise 2.1: Train DGM value network ===")

    X_val, y_val = generate_value_data(
        lqr=lqr,
        n_samples=20000,
        T=T,
        low=-3.0,
        high=3.0,
        seed=1234,
        device=device
    )

    value_model = NetDGM(
        input_dim=3,
        hidden_dim=100,
        output_dim=1,
        n_layers=3
    )

    value_model, value_loss_history = train_supervised_model(
        model=value_model,
        X=X_val,
        y=y_val,
        batch_size=256,
        lr=1e-3,
        epochs=200,
        device=device,
        print_every=20
    )

    plot_loss(
        value_loss_history,
        title="Exercise 2.1: DGM training loss for value function",
        save_path="outputs/ex2_1_value_loss.png"
    )

    evaluate_value_model(value_model, lqr, n_test=5, T=T, device=device)

    torch.save(value_model.state_dict(), "outputs/value_model_dgm.pt")
    print("Saved value model to outputs/value_model_dgm.pt")

    # --------------------------------------------------------
    # Exercise 2.2: supervised learning of Markov control
    # --------------------------------------------------------
    print("\n=== Exercise 2.2: Train FFN control network ===")

    X_ctrl, y_ctrl = generate_control_data(
        lqr=lqr,
        n_samples=20000,
        T=T,
        low=-3.0,
        high=3.0,
        seed=5678,
        device=device
    )

    control_model = FFN(
        layer_sizes=[3, 100, 100, 2],
        activation=nn.Tanh
    )

    control_model, control_loss_history = train_supervised_model(
        model=control_model,
        X=X_ctrl,
        y=y_ctrl,
        batch_size=256,
        lr=1e-3,
        epochs=200,
        device=device,
        print_every=20
    )

    plot_loss(
        control_loss_history,
        title="Exercise 2.2: FFN training loss for Markov control",
        save_path="outputs/ex2_2_control_loss.png"
    )

    evaluate_control_model(control_model, lqr, n_test=5, T=T, device=device)

    torch.save(control_model.state_dict(), "outputs/control_model_ffn.pt")
    print("Saved control model to outputs/control_model_ffn.pt")

    print("\nDone.")
    print("Saved figures:")
    print("  outputs/ex2_1_value_loss.png")
    print("  outputs/ex2_2_control_loss.png")


if __name__ == "__main__":
    main()