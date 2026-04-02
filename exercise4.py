import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from exercise1 import LQRProblem


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)


# =========================================================
# 1. Problem parameters
# =========================================================
T = 1.0

H_np = np.array([[0.1, 0.0],
                 [0.0, 0.2]])

M_np = np.array([[1.0, 0.0],
                 [0.0, 1.0]])

sigma_np = np.array([[0.3, 0.0],
                     [0.0, 0.2]])

C_np = np.array([[1.0, 0.0],
                 [0.0, 1.0]])

D_np = np.array([[1.0, 0.0],
                 [0.0, 1.0]])

R_np = np.array([[1.0, 0.0],
                 [0.0, 1.0]])

H = torch.tensor(H_np, dtype=torch.float32, device=device)
M = torch.tensor(M_np, dtype=torch.float32, device=device)
sigma = torch.tensor(sigma_np, dtype=torch.float32, device=device)
C = torch.tensor(C_np, dtype=torch.float32, device=device)
D = torch.tensor(D_np, dtype=torch.float32, device=device)
R = torch.tensor(R_np, dtype=torch.float32, device=device)


# =========================================================
# 2. output folder
# =========================================================
OUTPUT_DIR = "exercise4_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# 3. benchmark from Exercise 1
# =========================================================
lqr_solver = LQRProblem(
    H=H_np,
    M=M_np,
    sigma=sigma_np,
    C=C_np,
    D=D_np,
    R=R_np,
    T=T,
    device=device,
    dtype=torch.float32
)

time_grid = np.linspace(0.0, T, 2001)
lqr_solver.solve_riccati(time_grid)


# =========================================================
# 4. networks
# =========================================================
class FFN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=100, num_layers=3):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, t, x):
        inp = torch.cat([t, x], dim=1)
        return self.net(inp)


value_net = FFN(in_dim=3, out_dim=1, hidden_dim=100, num_layers=3).to(device)
policy_net = FFN(in_dim=3, out_dim=2, hidden_dim=100, num_layers=3).to(device)


# =========================================================
# 5. sampling
# =========================================================
def sample_interior(n_batch):
    t = torch.rand(n_batch, 1, device=device) * T
    x = -3.0 + 6.0 * torch.rand(n_batch, 2, device=device)
    t.requires_grad_(True)
    x.requires_grad_(True)
    return t, x

def sample_terminal(n_batch):
    t = torch.ones(n_batch, 1, device=device) * T
    x = -3.0 + 6.0 * torch.rand(n_batch, 2, device=device)
    t.requires_grad_(True)
    x.requires_grad_(True)
    return t, x


# =========================================================
# 6. autograd helpers
# =========================================================
def gradient(outputs, inputs):
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=True,
        retain_graph=True
    )[0]

def hessian_trace_sigma_sigmaT(u, x, sigma):
    grad_u = gradient(u, x)
    second_derivs = []

    for i in range(x.shape[1]):
        g_i = grad_u[:, i:i+1]
        h_i = torch.autograd.grad(
            g_i,
            x,
            grad_outputs=torch.ones_like(g_i),
            create_graph=True,
            retain_graph=True
        )[0]
        second_derivs.append(h_i)

    Hess = torch.stack(second_derivs, dim=1)
    SigmaSigmaT = sigma @ sigma.T
    tr_term = torch.einsum("ij,nij->n", SigmaSigmaT, Hess).unsqueeze(1)
    return tr_term


# =========================================================
# 7. quadratic forms
# =========================================================
def quad_form(x, A):
    return torch.sum((x @ A) * x, dim=1, keepdim=True)

def control_quad(a, D):
    return torch.sum((a @ D) * a, dim=1, keepdim=True)


# =========================================================
# 8. PDE residual
# =========================================================
def pde_residual(value_net, policy_net, t, x):
    u = value_net(t, x)
    u_t = gradient(u, t)
    u_x = gradient(u, x)

    diff_term = 0.5 * hessian_trace_sigma_sigmaT(u, x, sigma)

    Hx = x @ H.T
    a = policy_net(t, x)
    Ma = a @ M.T

    drift_term = torch.sum(u_x * Hx, dim=1, keepdim=True)
    control_term = torch.sum(u_x * Ma, dim=1, keepdim=True)
    running_cost = quad_form(x, C) + control_quad(a, D)

    return u_t + diff_term + drift_term + control_term + running_cost

def terminal_loss(value_net, tT, xT):
    uT = value_net(tT, xT)
    target = quad_form(xT, R)
    return torch.mean((uT - target) ** 2)


# =========================================================
# 9. Hamiltonian
# =========================================================
def hamiltonian(value_net, policy_net, t, x):
    v = value_net(t, x)
    v_x = gradient(v, x)

    Hx = x @ H.T
    a = policy_net(t, x)
    Ma = a @ M.T

    term1 = torch.sum(v_x * Hx, dim=1, keepdim=True)
    term2 = torch.sum(v_x * Ma, dim=1, keepdim=True)
    term3 = quad_form(x, C)
    term4 = control_quad(a, D)

    return term1 + term2 + term3 + term4


# =========================================================
# 10. training
# =========================================================
def train_value_net(value_net, policy_net, epochs=300, batch_size=128, lr=1e-3):
    optimizer = optim.Adam(value_net.parameters(), lr=lr)
    losses = []

    for ep in range(epochs):
        t, x = sample_interior(batch_size)
        tT, xT = sample_terminal(batch_size)

        optimizer.zero_grad()

        res = pde_residual(value_net, policy_net, t, x)
        loss_pde = torch.mean(res ** 2)
        loss_bc = terminal_loss(value_net, tT, xT)
        loss = loss_pde + loss_bc

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (ep + 1) % 50 == 0:
            print(f"[Value] Epoch {ep+1:4d} | Loss = {loss.item():.6e}")

    return losses

def train_policy_net(value_net, policy_net, epochs=200, batch_size=128, lr=1e-3):
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    losses = []

    for p in value_net.parameters():
        p.requires_grad = False

    for ep in range(epochs):
        t, x = sample_interior(batch_size)

        optimizer.zero_grad()

        H_val = hamiltonian(value_net, policy_net, t, x)
        loss = torch.mean(H_val)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (ep + 1) % 50 == 0:
            print(f"[Policy] Epoch {ep+1:4d} | Loss = {loss.item():.6e}")

    for p in value_net.parameters():
        p.requires_grad = True

    return losses


# =========================================================
# 11. evaluation against Exercise 1
# =========================================================
@torch.no_grad()
def evaluate_against_lqr(value_net, policy_net, lqr_solver, n_test=1000):
    t = torch.rand(n_test, 1, device=device) * T
    x = -3.0 + 6.0 * torch.rand(n_test, 2, device=device)

    t_for_lqr = t.squeeze(1)
    x_for_lqr = x.unsqueeze(1)

    v_true = lqr_solver.value_function(t_for_lqr, x_for_lqr)
    a_true = lqr_solver.markov_control(t_for_lqr, x_for_lqr)

    v_pred = value_net(t, x)
    a_pred = policy_net(t, x)

    value_mse = torch.mean((v_pred - v_true) ** 2).item()
    control_mse = torch.mean((a_pred - a_true) ** 2).item()

    return value_mse, control_mse


# =========================================================
# 12. slice plots
# =========================================================
@torch.no_grad()
def save_value_slice(value_net, lqr_solver, t_fixed=0.0, filename="value_slice.png"):
    x1 = torch.linspace(-3, 3, 200, device=device)
    x2 = torch.zeros_like(x1)
    x = torch.stack([x1, x2], dim=1)
    t = torch.full((200, 1), t_fixed, device=device)

    v_pred = value_net(t, x).squeeze().cpu().numpy()
    v_true = lqr_solver.value_function(
        t.squeeze(1),
        x.unsqueeze(1)
    ).squeeze().cpu().numpy()

    plt.figure(figsize=(7, 5))
    plt.plot(x1.cpu().numpy(), v_true, label="Exercise 1 benchmark")
    plt.plot(x1.cpu().numpy(), v_pred, "--", label="PIA + DGM")
    plt.xlabel("x1, with x2 = 0")
    plt.ylabel("Value")
    plt.title(f"Value comparison at t = {t_fixed}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close()

@torch.no_grad()
def save_control_slice(policy_net, lqr_solver, t_fixed=0.0, filename="control_slice.png"):
    x1 = torch.linspace(-3, 3, 200, device=device)
    x2 = torch.zeros_like(x1)
    x = torch.stack([x1, x2], dim=1)
    t = torch.full((200, 1), t_fixed, device=device)

    a_pred = policy_net(t, x).cpu().numpy()
    a_true = lqr_solver.markov_control(
        t.squeeze(1),
        x.unsqueeze(1)
    ).cpu().numpy()

    plt.figure(figsize=(7, 5))
    plt.plot(x1.cpu().numpy(), a_true[:, 0], label="True control dim 1")
    plt.plot(x1.cpu().numpy(), a_pred[:, 0], "--", label="Pred control dim 1")
    plt.plot(x1.cpu().numpy(), a_true[:, 1], label="True control dim 2")
    plt.plot(x1.cpu().numpy(), a_pred[:, 1], "--", label="Pred control dim 2")
    plt.xlabel("x1, with x2 = 0")
    plt.ylabel("Control")
    plt.title(f"Control comparison at t = {t_fixed}")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close()


# =========================================================
# 13. main loop
# =========================================================
def main():
    num_outer_iterations = 4

    all_value_train_losses = []
    all_policy_train_losses = []
    value_errors = []
    control_errors = []

    for k in range(num_outer_iterations):
        print(f"\n========== Policy Iteration Step {k+1}/{num_outer_iterations} ==========")

        value_losses = train_value_net(
            value_net=value_net,
            policy_net=policy_net,
            epochs=300,
            batch_size=128,
            lr=1e-3
        )
        all_value_train_losses.extend(value_losses)

        policy_losses = train_policy_net(
            value_net=value_net,
            policy_net=policy_net,
            epochs=200,
            batch_size=128,
            lr=1e-3
        )
        all_policy_train_losses.extend(policy_losses)

        v_err, a_err = evaluate_against_lqr(value_net, policy_net, lqr_solver, n_test=1000)
        value_errors.append(v_err)
        control_errors.append(a_err)

        print(f"After iteration {k+1}: value MSE = {v_err:.6e}, control MSE = {a_err:.6e}")

    plt.figure(figsize=(7, 5))
    plt.plot(all_value_train_losses)
    plt.yscale("log")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("Value network training loss")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "value_training_loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(all_policy_train_losses)
    plt.xlabel("Training step")
    plt.ylabel("Hamiltonian loss")
    plt.title("Policy network training loss")
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "policy_training_loss.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(range(1, num_outer_iterations + 1), value_errors, marker='o', label='Value MSE')
    plt.plot(range(1, num_outer_iterations + 1), control_errors, marker='s', label='Control MSE')
    plt.yscale("log")
    plt.xlabel("Policy iteration step")
    plt.ylabel("Error")
    plt.title("Convergence against Exercise 1 benchmark")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_convergence.png"), dpi=200, bbox_inches="tight")
    plt.close()

    save_value_slice(value_net, lqr_solver, t_fixed=0.0, filename="value_slice_t0.png")
    save_control_slice(policy_net, lqr_solver, t_fixed=0.0, filename="control_slice_t0.png")

    print(f"\nAll figures saved in folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()