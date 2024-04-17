import torch
import torch.nn as nn

from solver.lp_sparse_forward_diff import ODESYSLP


class ODEINDLayer(nn.Module):
    """ class for ODE with dimensions modeled independently"""
    def __init__(self, bs, order, n_ind_dim, n_iv, n_step, n_iv_steps, solver_dbl=True, gamma=0.5, alpha=0.1, central_diff=True, double_ret=False, device=None):
        super().__init__()
        # placeholder step size
        self.step_size = 0.1
        self.n_step = n_step  # int(self.end / self.step_size)
        self.order = order

        self.n_ind_dim = n_ind_dim
        self.n_dim = 1  # n_dim
        self.n_equations =1 # n_equations
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
            bs=bs*self.n_ind_dim, n_dim=self.n_dim, n_equations=self.n_equations, n_auxiliary=0, n_step=self.n_step,
            step_size=self.step_size, order=self.order, n_iv=self.n_iv, n_iv_steps=self.n_iv_steps, dtype=dtype,
            device=self.device,
        )

        self.gamma_alpha = gamma * alpha

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

        self.ode.build_ode(coeffs, rhs, iv_rhs, derivative_constraints)
        A = self.ode.AG.to_dense()
        beta = self.ode.ub.to(dtype=A.dtype)

        A = A[..., 1:]
        At = A.transpose(-2, -1)
        AtA = At @ A
        L, info = torch.linalg.cholesky_ex(AtA, upper=False, check_errors=False)
        rhs = At @ beta[..., None]
        # rhs[..., 0, :] += self.gamma_alpha
        x = rhs.cholesky_solve(L, upper=False)[..., 0]

        # eps = x[:, 0]
        eps = torch.zeros_like(x[:, 0])

        # shape: batch, step, vars (== 1), order
        u = self.ode.get_solution_reshaped(x)

        u = u.reshape(self.bs, self.n_ind_dim, self.n_step, self.order+1)
        # shape: batch, step, vars, order

        u0 = u[:, :, :, 0]
        u1 = u[:, :, :, 1]
        u2 = u[:, :, :, 2]

        return u0, u1, u2, eps, steps


class ODESYSLayer(nn.Module):
    def __init__(self, bs, order, n_ind_dim, n_dim, n_equations, n_iv, n_iv_steps, n_step, periodic_boundary=False, solver_dbl=True, gamma=0.5, alpha=0.1, double_ret=True, device=None):
        super().__init__()

        # placeholder step size
        self.step_size = 0.1
        self.n_step = n_step 
        self.order = order

        self.n_dim = n_dim
        self.n_ind_dim = n_ind_dim
        self.n_equations = n_equations
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

        dtype = torch.float64 if solver_dbl else torch.float32

        self.ode = ODESYSLP(bs=bs*self.n_ind_dim, n_dim=self.n_dim, n_equations=n_equations, n_auxiliary=0, n_step=self.n_step, step_size=self.step_size, order=self.order,
                        periodic_boundary=periodic_boundary, n_iv=self.n_iv, n_iv_steps=self.n_iv_steps, dtype=dtype, device=self.device)

        self.gamma_alpha = gamma * alpha

    def forward(self, coeffs, rhs, iv_rhs, steps):
        coeffs = coeffs.reshape(self.bs*self.n_ind_dim, self.n_equations, self.n_step,self.n_dim, self.order + 1)

        #n_equation, n_step
        rhs = rhs.reshape(self.bs*self.n_ind_dim, self.n_equations*self.n_step)

        #iv_steps, n_dim, n_iv
        if iv_rhs is not None:
            iv_rhs = iv_rhs.reshape(self.bs*self.n_ind_dim, -1)

        steps = steps.reshape(self.bs * self.n_ind_dim, self.n_step - 1, self.n_dim)

        if self.solver_dbl:
            coeffs = coeffs.double()
            rhs = rhs.double()
            iv_rhs = iv_rhs.double() if iv_rhs is not None else None
            steps = steps.double()

        derivative_constraints = self.ode.build_derivative_tensor(steps)
        eq_constraints = self.ode.build_equation_tensor(coeffs)

        x = self.qpf(eq_constraints, rhs, iv_rhs, derivative_constraints)

        eps = x[:, 0]

        #shape: batch, step, vars (== 1), order
        u = self.ode.get_solution_reshaped(x)

        u = u.reshape(self.bs,self.n_ind_dim, self.n_step, self.n_dim, self.order+1)
        #shape: batch, step, vars, order

        u0 = u[:, :, :, :, 0]
        u1 = u[:, :, :, :, 1]
        u2 = u[:, :, :, :, 2]
        
        return u0, u1, u2, eps, u
