import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


class LQRProblem:
    """
    Solve a 2D LQR problem:
        dX_s = (H X_s + M a_s) ds + sigma dW_s
    with running cost:
        X_s^T C X_s + a_s^T D a_s
    and terminal cost:
        X_T^T R X_T

    The value function has form:
        v(t,x) = x^T S(t) x + integral_t^T tr(sigma sigma^T S(r)) dr
    where S solves the Riccati ODE.
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
        self.g_grid = None  # stores integral term on grid

    def _riccati_rhs(self, t, y):
        """
        y is flattened 2x2 matrix S(t).
        ODE:
            S'(t) = -2 H^T S + S M D^{-1} M^T S - C
        """
        S = y.reshape(2, 2)
        rhs = -2.0 * self.H.T @ S + S @ self.M @ self.D_inv @ self.M.T @ S - self.C
        return rhs.reshape(-1)

    def solve_riccati(self, time_grid):
        """
        Solve Riccati ODE backward from T to 0 on a specified increasing time grid.
        Also compute:
            g(t) = integral_t^T tr(sigma sigma^T S(r)) dr
        using trapezoidal rule on the same grid.
        """
        time_grid = np.asarray(time_grid, dtype=float)
        assert np.all(np.diff(time_grid) > 0), "time_grid must be strictly increasing"
        assert abs(time_grid[-1] - self.T) < 1e-12, "time_grid must end at T"

        yT = self.R.reshape(-1)

        # solve backward from T to 0
        sol = solve_ivp(
            fun=self._riccati_rhs,
            t_span=(self.T, time_grid[0]),
            y0=yT,
            t_eval=time_grid[::-1],   # decreasing order because backward solve
            rtol=1e-9,
            atol=1e-11,
            method="RK45"
        )

        if not sol.success:
            raise RuntimeError("Riccati ODE solver failed.")

        # reverse back to increasing time order
        S_rev = sol.y.T[::-1]   # shape (N, 4)
        S_grid = S_rev.reshape(-1, 2, 2)

        # symmetrize numerically
        S_grid = 0.5 * (S_grid + np.transpose(S_grid, (0, 2, 1)))

        # compute integral term g(t) = integral_t^T tr(sigma sigma^T S(r)) dr
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
        """
        Piecewise linear interpolation of S(t) and g(t).
        t_values: numpy array shape (batch,)
        """
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

                S_out[i] = (1 - w) * self.S_grid[j] + w * self.S_grid[j + 1]
                g_out[i] = (1 - w) * self.g_grid[j] + w * self.g_grid[j + 1]

        return S_out, g_out

    def value_function(self, t_batch, x_batch):
        """
        Input:
            t_batch: torch tensor of shape (batch,)
            x_batch: torch tensor of shape (batch, 1, 2)

        Output:
            values: torch tensor of shape (batch, 1)
        """
        t_np = t_batch.detach().cpu().numpy()
        x_np = x_batch.detach().cpu().numpy().reshape(-1, 2)

        S_np, g_np = self._interp_S_and_g(t_np)

        vals = np.einsum("bi,bij,bj->b", x_np, S_np, x_np) + g_np
        vals = vals.reshape(-1, 1)

        return torch.tensor(vals, dtype=self.dtype, device=self.device)

    def markov_control(self, t_batch, x_batch):
        """
        Input:
            t_batch: torch tensor of shape (batch,)
            x_batch: torch tensor of shape (batch, 1, 2)

        Output:
            controls: torch tensor of shape (batch, 2)
        """
        t_np = t_batch.detach().cpu().numpy()
        x_np = x_batch.detach().cpu().numpy().reshape(-1, 2)

        S_np, _ = self._interp_S_and_g(t_np)

        controls = np.zeros((len(t_np), 2))
        for i in range(len(t_np)):
            controls[i] = - self.D_inv @ self.M.T @ S_np[i] @ x_np[i]

        return torch.tensor(controls, dtype=self.dtype, device=self.device)
    
# Example placeholder matrices
H = np.array([[0.1, 0.0],
              [0.0, 0.2]])

M = np.array([[1.0, 0.0],
              [0.0, 1.0]])

sigma = np.array([[0.3, 0.0],
                  [0.0, 0.2]])

C = np.array([[1.0, 0.0],
              [0.0, 1.0]])

D = np.array([[1.0, 0.0],
              [0.0, 1.0]])

R = np.array([[1.0, 0.0],
              [0.0, 1.0]])

T = 1.0

lqr = LQRProblem(H, M, sigma, C, D, R, T)

time_grid = np.linspace(0.0, T, 2001)
lqr.solve_riccati(time_grid)

# test batches
t_batch = torch.tensor([0.0, 0.25, 0.5], dtype=torch.float32)
x_batch = torch.tensor([[[1.0, 2.0]],
                        [[-1.0, 0.5]],
                        [[0.2, -0.3]]], dtype=torch.float32)

v = lqr.value_function(t_batch, x_batch)
a = lqr.markov_control(t_batch, x_batch)

print("value function:")
print(v)

print("controls:")
print(a)

def simulate_lqr_cost_explicit(
    lqr,
    x0,
    t0=0.0,
    N=100,
    n_paths=10000,
    seed=1234
):
    """
    Simulate the optimally controlled SDE using explicit Euler.

    Returns:
        mc_mean_cost, mc_std_cost
    """
    rng = np.random.default_rng(seed)

    dt = (lqr.T - t0) / N
    time_grid = np.linspace(t0, lqr.T, N + 1)

    x0 = np.asarray(x0, dtype=float)
    X = np.tile(x0, (n_paths, 1))   # shape (n_paths, 2)

    running_cost = np.zeros(n_paths)

    sqrt_dt = np.sqrt(dt)

    for n in range(N):
        t_n = np.full(n_paths, time_grid[n], dtype=float)

        # control a(t_n, X_n)
        t_torch = torch.tensor(t_n, dtype=torch.float32)
        x_torch = torch.tensor(X.reshape(n_paths, 1, 2), dtype=torch.float32)

        a = lqr.markov_control(t_torch, x_torch).detach().cpu().numpy()  # (n_paths, 2)

        # running cost approximation
        xCx = np.einsum("bi,ij,bj->b", X, lqr.C, X)
        aDa = np.einsum("bi,ij,bj->b", a, lqr.D, a)
        running_cost += (xCx + aDa) * dt

        # Euler step
        dW = rng.normal(size=(n_paths, 2)) * sqrt_dt
        drift = (X @ lqr.H.T) + (a @ lqr.M.T)
        diffusion = dW @ lqr.sigma.T

        X = X + drift * dt + diffusion

    terminal_cost = np.einsum("bi,ij,bj->b", X, lqr.R, X)
    total_cost = running_cost + terminal_cost

    return total_cost.mean(), total_cost.std(ddof=1)

def compute_mc_error(lqr, x0, t0, N, n_paths, seed=1234):
    mc_mean, mc_std = simulate_lqr_cost_explicit(
        lqr=lqr,
        x0=x0,
        t0=t0,
        N=N,
        n_paths=n_paths,
        seed=seed
    )

    t_batch = torch.tensor([t0], dtype=torch.float32)
    x_batch = torch.tensor([[x0]], dtype=torch.float32)
    true_value = lqr.value_function(t_batch, x_batch).item()

    error = abs(mc_mean - true_value)
    return error, mc_mean, mc_std, true_value

x0 = np.array([1.0, -1.0])
t0 = 0.0
n_paths_large = 100000

N_list = [1, 10, 50, 100, 500, 1000, 5000]
errors_time = []

for N in N_list:
    err, mc_mean, mc_std, true_val = compute_mc_error(
        lqr=lqr,
        x0=x0,
        t0=t0,
        N=N,
        n_paths=n_paths_large,
        seed=1234
    )
    errors_time.append(err)
    print(f"N={N:<5d} error={err:.6e}, MC={mc_mean:.6f}, true={true_val:.6f}")

plt.figure(figsize=(7, 5))
plt.loglog(N_list, errors_time, marker='o')
plt.xlabel("Number of time steps N")
plt.ylabel("Absolute error")
plt.title("Monte Carlo error vs time steps")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.show()

N_large = 5000
mc_list = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
errors_mc = []

for n_paths in mc_list:
    err, mc_mean, mc_std, true_val = compute_mc_error(
        lqr=lqr,
        x0=x0,
        t0=t0,
        N=N_large,
        n_paths=n_paths,
        seed=1234
    )
    errors_mc.append(err)
    print(f"MC={n_paths:<6d} error={err:.6e}, MC={mc_mean:.6f}, true={true_val:.6f}")

plt.figure(figsize=(7, 5))
plt.loglog(mc_list, errors_mc, marker='o')
plt.xlabel("Number of Monte Carlo samples")
plt.ylabel("Absolute error")
plt.title("Monte Carlo error vs sample size")
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.show()