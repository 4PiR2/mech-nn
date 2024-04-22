import torch
from torch import nn

from solver.ode_forward import ode_forward


class ODEINDLayer(nn.Module):
    """ class for ODE with dimensions modeled independently"""
    def __init__(self, bs, order, n_ind_dim, n_iv, n_step, n_iv_steps, solver_dbl=True, gamma=0.5, alpha=0.1, central_diff=True, double_ret=False, device=None):
        super().__init__()
        self.n_step = n_step
        self.order = order
        self.n_ind_dim = n_ind_dim
        self.n_iv = n_iv
        self.n_iv_steps = n_iv_steps
        self.bs = bs
        self.solver_dbl = solver_dbl

    def forward(self, coeffs, rhs, iv_rhs, steps):
        coeffs = coeffs.reshape(self.bs, self.n_ind_dim, self.n_step, 1, 1, self.order + 1)
        rhs = rhs.reshape(self.bs, self.n_ind_dim, self.n_step, 1)
        if iv_rhs is not None:
            iv_rhs = iv_rhs.reshape(self.bs, self.n_ind_dim, self.n_iv_steps, 1, self.n_iv)
        else:
            iv_rhs = torch.empty(self.bs, self.n_ind_dim, 0, 1, self.n_iv, device=coeffs.device)
        steps = steps.reshape(self.bs, self.n_ind_dim, self.n_step - 1)

        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double()
            steps = steps.double()

        u = ode_forward(coeffs, rhs, iv_rhs, steps)[..., 0, :]
        u0, u1, u2, *_ = u.unbind(dim=-1)
        return u0, u1, u2, torch.zeros(()), steps.reshape(self.bs * self.n_ind_dim, self.n_step - 1, 1)


class ODESYSLayer(nn.Module):
    def __init__(self, bs, order, n_ind_dim, n_dim, n_equations, n_iv, n_iv_steps, n_step, periodic_boundary=False, solver_dbl=True, gamma=0.5, alpha=0.1, double_ret=True, device=None):
        super().__init__()
        self.n_step = n_step
        self.order = order
        self.n_dim = n_dim
        self.n_ind_dim = n_ind_dim
        self.n_equations = n_equations
        self.n_iv = n_iv
        self.n_iv_steps = n_iv_steps
        self.bs = bs
        self.solver_dbl = solver_dbl

    def forward(self, coeffs, rhs, iv_rhs, steps):
        coeffs = coeffs.reshape(self.bs, self.n_ind_dim, self.n_equations, self.n_step, self.n_dim, self.order + 1).transpose(-4, -3)
        rhs = rhs.reshape(self.bs, self.n_ind_dim, self.n_equations, self.n_step).transpose(-2, -1)
        if iv_rhs is not None:
            iv_rhs = iv_rhs.reshape(self.bs, self.n_ind_dim, self.n_iv_steps, self.n_dim, self.n_iv)
        else:
            iv_rhs = torch.empty(self.bs, self.n_ind_dim, 0, self.n_dim, self.n_iv, device=coeffs.device)
        steps = steps.reshape(self.bs, self.n_ind_dim, self.n_step - 1, self.n_dim).mean(dim=-1)

        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double()
            steps = steps.double()

        u = ode_forward(coeffs, rhs, iv_rhs, steps)
        u0, u1, u2, *_ = u.unbind(dim=-1)
        return u0, u1, u2, torch.zeros(()), u
