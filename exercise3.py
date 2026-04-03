import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


# ============================================================
# 0. Utilities
# ============================================================

def set_seed(seed: int = 1234):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 1. Exercise 1.1 benchmark solver (kept here for convenience)
# ============================================================

class LQRProblem:
    """
    Benchmark LQR class from Exercise 1.1.

    This is included here because it is useful for:
    - reusing the same matrices H, M, sigma, C, D, R, T
    - comparing with benchmark quantities if needed
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
        t_np = t_batch.detach().cpu().numpy()
        x_np = x_batch.detach().cpu().numpy().reshape(-1, 2)

        S_np, g_np = self._interp_S_and_g(t_np)
        vals = np.einsum("bi,bij,bj->b", x_np, S_np, x_np) + g_np
        vals = vals.reshape(-1, 1)

        return torch.tensor(vals, dtype=self.dtype, device=self.device)

    def markov_control(self, t_batch, x_batch):
        t_np = t_batch.detach().cpu().numpy()
        x_np = x_batch.detach().cpu().numpy().reshape(-1, 2)

        S_np, _ = self._interp_S_and_g(t_np)
        controls = np.zeros((len(t_np), 2))

        for i in range(len(t_np)):
            controls[i] = - self.D_inv @ self.M.T @ S_np[i] @ x_np[i]

        return torch.tensor(controls, dtype=self.dtype, device=self.device)


# ============================================================
# 2. DGM network
# ============================================================

class DGMLayer(nn.Module):
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
    Input:
        X = [t, x1, x2], shape (batch, 3)
    Output:
        u(t,x), shape (batch, 1)
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

    def forward(self, X):
        s = self.input_layer(X)
        for layer in self.dgm_layers:
            s = layer(X, s)
        return self.output_layer(s)


# ============================================================
# 3. Monte Carlo under constant control alpha=(1,1)
# ============================================================

def simulate_constant_control_cost(
    H, M, sigma, C, D, R, T,
    x0,
    t0=0.0,
    alpha=np.array([1.0, 1.0]),
    N=1000,
    n_paths=20000,
    seed=1234
):
    """
    Monte Carlo estimator for the linear PDE solution under constant control alpha.

    Dynamics:
        dX_s = (H X_s + M alpha) ds + sigma dW_s

    Cost:
        E [ integral_t^T (X^T C X + alpha^T D alpha) ds + X_T^T R X_T ]
    """
    rng = np.random.default_rng(seed)

    dt = (T - t0) / N
    sqrt_dt = np.sqrt(dt)

    x0 = np.asarray(x0, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    X = np.tile(x0, (n_paths, 1))
    running_cost = np.zeros(n_paths)

    alpha_D_alpha = alpha @ D @ alpha

    for _ in range(N):
        xCx = np.einsum("bi,ij,bj->b", X, C, X)
        running_cost += (xCx + alpha_D_alpha) * dt

        dW = rng.normal(size=(n_paths, 2)) * sqrt_dt
        drift = X @ H.T + np.tile(alpha @ M.T, (n_paths, 1))
        diffusion = dW @ sigma.T

        X = X + drift * dt + diffusion

    terminal_cost = np.einsum("bi,ij,bj->b", X, R, X)
    total_cost = running_cost + terminal_cost

    return total_cost.mean(), total_cost.std(ddof=1)


# ============================================================
# 4. DGM solver for Exercise 3
# ============================================================

class LinearPDEDGM:
    """
    Solve the Exercise 3 linear PDE using DGM.

    PDE:
        u_t
        + 1/2 tr(sigma sigma^T Hess_x u)
        + grad_x u^T H x
        + grad_x u^T M alpha
        + x^T C x
        + alpha^T D alpha
        = 0

    Terminal condition:
        u(T,x) = x^T R x
    """

    def __init__(
        self,
        H, M, sigma, C, D, R, T,
        alpha=np.array([1.0, 1.0]),
        hidden_dim=100,
        n_layers=3,
        device="cpu"
    ):
        self.H = torch.tensor(H, dtype=torch.float32, device=device)
        self.M = torch.tensor(M, dtype=torch.float32, device=device)
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
        self.C = torch.tensor(C, dtype=torch.float32, device=device)
        self.D = torch.tensor(D, dtype=torch.float32, device=device)
        self.R = torch.tensor(R, dtype=torch.float32, device=device)
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=device)

        self.T = float(T)
        self.device = device

        self.model = NetDGM(
            input_dim=3,
            hidden_dim=hidden_dim,
            output_dim=1,
            n_layers=n_layers
        ).to(device)

    def sample_interior(self, batch_size, x_low=-3.0, x_high=3.0):
        t = torch.rand(batch_size, 1, device=self.device) * self.T
        x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=self.device)
        t.requires_grad_(True)
        x.requires_grad_(True)
        return t, x

    def sample_terminal(self, batch_size, x_low=-3.0, x_high=3.0):
        t = torch.full((batch_size, 1), self.T, device=self.device)
        x = x_low + (x_high - x_low) * torch.rand(batch_size, 2, device=self.device)
        t.requires_grad_(True)
        x.requires_grad_(True)
        return t, x

    def net_u(self, t, x):
        X = torch.cat([t, x], dim=1)
        return self.model(X)

    def grad_u(self, u, inputs):
        return torch.autograd.grad(
            outputs=u,
            inputs=inputs,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True
        )[0]

    def hessian_x(self, u, x):
        """
        Returns Hessian matrix wrt x, shape (batch, 2, 2)
        """
        grad = self.grad_u(u, x)  # (batch, 2)
        batch_size = x.shape[0]
        hess = []

        for i in range(2):
            grad_i = grad[:, i:i+1]
            second = torch.autograd.grad(
                outputs=grad_i,
                inputs=x,
                grad_outputs=torch.ones_like(grad_i),
                create_graph=True,
                retain_graph=True
            )[0]  # (batch, 2)
            hess.append(second)

        # hess[0] = d/dx grad[:,0], hess[1] = d/dx grad[:,1]
        # stack as rows
        Hx = torch.stack(hess, dim=1)  # (batch, 2, 2)
        return Hx

    def pde_residual(self, t, x):
        """
        Compute PDE residual at interior points.
        """
        u = self.net_u(t, x)                    # (batch,1)
        u_t = self.grad_u(u, t)                 # (batch,1)
        grad_x = self.grad_u(u, x)              # (batch,2)
        hess_x = self.hessian_x(u, x)           # (batch,2,2)

        sigma_sigma_T = self.sigma @ self.sigma.T
        # trace(sigma sigma^T Hess u)
        diff_term = 0.5 * torch.einsum("ij,bij->b", sigma_sigma_T, hess_x).unsqueeze(1)

        Hx = x @ self.H.T                       # (batch,2)
        Malpha = torch.matmul(self.alpha, self.M.T).unsqueeze(0).expand(x.shape[0], -1)

        drift_term1 = torch.sum(grad_x * Hx, dim=1, keepdim=True)
        drift_term2 = torch.sum(grad_x * Malpha, dim=1, keepdim=True)

        xCx = torch.einsum("bi,ij,bj->b", x, self.C, x).unsqueeze(1)
        alphaDalpha = torch.einsum("i,ij,j->", self.alpha, self.D, self.alpha).reshape(1, 1)
        alphaDalpha = alphaDalpha.expand(x.shape[0], 1)

        residual = u_t + diff_term + drift_term1 + drift_term2 + xCx + alphaDalpha
        return residual

    def boundary_loss(self, tT, xT):
        uT = self.net_u(tT, xT)
        target = torch.einsum("bi,ij,bj->b", xT, self.R, xT).unsqueeze(1)
        return torch.mean((uT - target) ** 2)

    def train(
        self,
        epochs=3000,
        batch_size=256,
        lr=1e-3,
        x_low=-3.0,
        x_high=3.0,
        print_every=100,
        eval_every=200,
        mc_x0_list=None,
        mc_N=1000,
        mc_paths=10000,
        mc_seed=1234
    ):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        loss_history = []
        eval_steps = []
        eval_errors = []

        H_np = self.H.detach().cpu().numpy()
        M_np = self.M.detach().cpu().numpy()
        sigma_np = self.sigma.detach().cpu().numpy()
        C_np = self.C.detach().cpu().numpy()
        D_np = self.D.detach().cpu().numpy()
        R_np = self.R.detach().cpu().numpy()
        alpha_np = self.alpha.detach().cpu().numpy()

        if mc_x0_list is None:
            mc_x0_list = [
                np.array([0.0, 0.0]),
                np.array([1.0, -1.0]),
                np.array([2.0, 1.5]),
            ]

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()

            # interior points
            t_int, x_int = self.sample_interior(batch_size, x_low, x_high)
            residual = self.pde_residual(t_int, x_int)
            eq_loss = torch.mean(residual ** 2)

            # terminal points
            t_T, x_T = self.sample_terminal(batch_size, x_low, x_high)
            bd_loss = self.boundary_loss(t_T, x_T)

            loss = eq_loss + bd_loss
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if epoch % print_every == 0:
                print(
                    f"Epoch {epoch:5d} | "
                    f"Loss = {loss.item():.6e} | "
                    f"EqLoss = {eq_loss.item():.6e} | "
                    f"BdLoss = {bd_loss.item():.6e}"
                )

            if epoch % eval_every == 0:
                # Compare network prediction against Monte Carlo at a few fixed points
                errs = []

                self.model.eval()
                with torch.no_grad():
                    for x0 in mc_x0_list:
                        mc_mean, _ = simulate_constant_control_cost(
                            H=H_np,
                            M=M_np,
                            sigma=sigma_np,
                            C=C_np,
                            D=D_np,
                            R=R_np,
                            T=self.T,
                            x0=x0,
                            t0=0.0,
                            alpha=alpha_np,
                            N=mc_N,
                            n_paths=mc_paths,
                            seed=mc_seed
                        )

                        X_input = torch.tensor(
                            np.array([[0.0, x0[0], x0[1]]]),
                            dtype=torch.float32,
                            device=self.device
                        )
                        pred = self.model(X_input).item()
                        errs.append(abs(pred - mc_mean))

                mean_err = float(np.mean(errs))
                eval_steps.append(epoch)
                eval_errors.append(mean_err)
                self.model.train()

                print(f"           MC comparison mean abs error = {mean_err:.6e}")

        return loss_history, eval_steps, eval_errors


# ============================================================
# 5. Plotting helpers
# ============================================================

def plot_training_loss(loss_history, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(loss_history)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Training loss")
    plt.title("Exercise 3.1: DGM training loss")
    plt.grid(True, alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


def plot_mc_error(eval_steps, eval_errors, save_path=None):
    plt.figure(figsize=(7, 5))
    plt.plot(eval_steps, eval_errors, marker="o")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Mean abs error vs Monte Carlo")
    plt.title("Exercise 3.1: Error against Monte Carlo solution")
    plt.grid(True, alpha=0.3)
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()


# ============================================================
# 6. Main
# ============================================================

def main():
    set_seed(1234)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    T = 1.0

    # Replace these with your Example 4.13 matrices if needed.
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

    alpha = np.array([1.0, 1.0])

    os.makedirs("outputs", exist_ok=True)

    solver = LinearPDEDGM(
        H=H, M=M, sigma=sigma, C=C, D=D, R=R, T=T,
        alpha=alpha,
        hidden_dim=100,
        n_layers=3,
        device=device
    )

    loss_history, eval_steps, eval_errors = solver.train(
        epochs=3000,
        batch_size=256,
        lr=1e-3,
        x_low=-3.0,
        x_high=3.0,
        print_every=100,
        eval_every=200,
        mc_x0_list=[
            np.array([0.0, 0.0]),
            np.array([1.0, -1.0]),
            np.array([2.0, 1.5]),
        ],
        mc_N=1000,
        mc_paths=10000,
        mc_seed=1234
    )

    plot_training_loss(loss_history, save_path="outputs/ex3_training_loss.png")
    plot_mc_error(eval_steps, eval_errors, save_path="outputs/ex3_mc_error.png")

    torch.save(solver.model.state_dict(), "outputs/ex3_dgm_model.pt")
    print("\nSaved files:")
    print("  outputs/ex3_training_loss.png")
    print("  outputs/ex3_mc_error.png")
    print("  outputs/ex3_dgm_model.pt")


if __name__ == "__main__":
    main()