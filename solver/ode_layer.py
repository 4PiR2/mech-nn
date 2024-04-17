import torch
import torch.nn as nn

from solver.lp_sparse_forward_diff import ODESYSLP


class ODEINDLayer(nn.Module):
    """ class for ODE with dimensions modeled independently"""
    def __init__(self, bs, order, n_ind_dim, n_iv, n_step, n_iv_steps, solver_dbl=True, gamma=0.5, alpha=0.1, central_diff=True, double_ret=False, device=None):
        super().__init__()
        # placeholder step size
        self.step_size = .1
        self.n_step = n_step  # int(self.end / self.step_size)
        self.order = order

        self.n_ind_dim = n_ind_dim
        self.n_dim = 1  # n_dim
        self.n_iv = n_iv
        self.n_iv_steps = n_iv_steps
        self.bs = bs
        self.n_coeff = self.n_step * (self.order + 1)
        self.device = device
        self.solver_dbl = solver_dbl

        if solver_dbl:
            print("Using double precision solver")
        else:
            print("Using single precision solver")

        dtype = torch.float64 if self.solver_dbl else torch.float32

        self.ode = ODESYSLP(
            batch_size=bs * self.n_ind_dim, n_dims=self.n_dim, n_steps=self.n_step,
            ode_order=self.order, n_init_orders=self.n_iv, n_init_var_steps=self.n_iv_steps, dtype=dtype,
            device=self.device,
        )

    def forward(self, coeffs, rhs, iv_rhs, steps):
        coeffs = coeffs.reshape(self.bs * self.n_ind_dim, self.n_step, self.n_dim, self.order + 1)

        rhs = rhs.reshape(self.bs * self.n_ind_dim, self.n_step)
        if iv_rhs is not None:
            iv_rhs = iv_rhs.reshape(self.bs * self.n_ind_dim, self.n_iv_steps * self.n_iv)

        steps = steps.reshape(self.bs * self.n_ind_dim, self.n_step-1,1)

        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps = steps.double()

        derivative_constraints = self.ode.build_derivative_tensor(steps)

        A = self.ode.build_ode(coeffs, derivative_constraints).to_dense()
        beta = torch.cat([rhs, iv_rhs], dim=1).to(dtype=A.dtype)

        A = A[..., 1:]
        At = A.transpose(-2, -1)
        AtA = At @ A
        L, info = torch.linalg.cholesky_ex(AtA, upper=False, check_errors=False)

        # from matplotlib import pyplot as plt
        # a = A[0].bool()
        # a = AtA[0].bool()
        # a = L[0].bool()
        # plt.imshow(a.detach().cpu().numpy())
        # plt.tight_layout()
        # plt.savefig('l1.png')
        # plt.show()

        rhs1 = At[..., :beta.size(-1)] @ beta[..., None]
        x = rhs1.cholesky_solve(L, upper=False)[..., 0]

        # shape: batch, step, vars (== 1), order
        u = x.reshape(self.bs, self.n_ind_dim, self.n_step, self.order+1)

        u0 = u[..., 0]
        u1 = u[..., 1]
        u2 = u[..., 2]
        eps = torch.zeros_like(x[:, 0])

        return u0, u1, u2, eps, steps
