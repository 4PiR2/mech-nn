import itertools
import math

import numpy as np
import torch
from torch import nn


class ODEINDLayer(nn.Module):
    """ class for ODE with dimensions modeled independently"""
    def __init__(
            self,
            bs: int = 1,  # batch_size
            n_ind_dim: int = 1,
            n_step: int = 3,  # n_steps
            n_dim: int = 1,  # n_dims
            order: int = 2,  # ode order
            n_equations: int = 1,
            n_iv_steps: int = 2,  # n_init_var_steps
            n_iv: int = 2,  # n_init_orders
            periodic_boundary: bool = False,
            solver_dbl: bool = True,
            gamma: float = .5,
            alpha: float = .1,
            central_diff: bool = True,
            double_ret: bool = False,
            device: torch.device = None,
    ):
        super().__init__()

        batch_size = bs * n_ind_dim
        n_steps = n_step
        n_dims = n_dim
        ode_order = order
        n_init_var_steps = n_iv_steps
        n_init_orders = n_iv
        dtype = torch.float64 if solver_dbl else torch.float32
        del bs, n_ind_dim, n_step, n_dim, order, n_iv_steps, n_iv, solver_dbl, gamma, alpha, central_diff, double_ret
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.n_dims = n_dims
        self.n_init_var_steps = n_init_var_steps
        self.n_init_orders = n_init_orders

        # order is diffeq order. n_order is total number of terms: y'', y', y for order 2.
        self.n_orders: int = ode_order + 1

        # total number of qp variables
        n_vars = n_dims * n_steps * self.n_orders + 1

        # Variables except eps. Used for raveling
        self.multi_index_shape = (n_steps, n_dims, self.n_orders)

        # build skeleton constraints, one equation for each dimension, filled during training
        equation_constraints = [
            [(step, dim, order) for dim, order in itertools.product(range(n_dims), range(self.n_orders))]
            for step, _ in itertools.product(range(n_steps), range(n_equations))
        ]
        constraint_indices = []
        for i, equation in enumerate(equation_constraints):
            for k in equation:
                constraint_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1])
        self.equation_indices = torch.cat([
            torch.arange(batch_size).repeat_interleave(len(constraint_indices))[None],
            torch.tensor(constraint_indices).t().repeat(1, batch_size),
        ], dim=-2)

        initial_constraints = [{
            (step, dim, order): 1.,
        } for step, dim, order in itertools.product(range(n_init_var_steps), range(n_dims), range(n_init_orders))]

        if periodic_boundary:
            initial_constraints += [{
                (0, dim, order): 1.,
                (n_steps - 1, dim, order): -1.,
            } for dim, order in itertools.product(range(n_dims), range(self.n_orders - 1))]

        constraint_indices = []
        constraint_values = []
        for i, equation in enumerate(initial_constraints):
            for k, v in equation.items():
                constraint_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1])
                constraint_values.append(v)
        initial_A = torch.sparse_coo_tensor(
            indices=torch.tensor(constraint_indices).t(),
            values=torch.tensor(constraint_values),
            size=[len(initial_constraints), n_vars],
            dtype=dtype,
            device=device
        )
        self.initial_A = torch.stack([initial_A] * batch_size, dim=0)  # (b, r1, c)

        EPS = 12345

        # forward
        smooth_constraints = [[
            EPS,
            (step + 1, dim, i),
            *[(step, dim, j) for j in range(i, self.n_orders)]
        ] for step, dim, i in itertools.product(range(n_steps - 1), range(n_dims), range(self.n_orders - 1))]
        # central
        smooth_constraints += [[
            EPS,
            (step - 1, dim, self.n_orders - 2),
            (step + 1, dim, self.n_orders - 2),
            (step, dim, self.n_orders - 1),
        ] for step, dim in itertools.product(range(1, n_steps - 1), range(n_dims))]
        # backward
        smooth_constraints += [[
            EPS,
            (step, dim, i),
            *[(step + 1, dim, j) for j in range(i, self.n_orders)]
        ] for step, dim, i in itertools.product(range(n_steps - 1), range(n_dims), range(self.n_orders - 1))]

        constraint_indices = []
        for i, equation in enumerate(smooth_constraints):
            for k in equation:
                constraint_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1] if k != EPS else [i, 0])
        self.smooth_indices = torch.cat([
            torch.arange(batch_size).repeat_interleave(len(constraint_indices))[None],
            torch.tensor(constraint_indices).t().repeat(1, batch_size),
        ], dim=-2)

    def forward(self, coeffs, rhs, iv_rhs, steps):
        """
        coeffs: (..., n_steps, n_equations, n_dims, n_orders)
        rhs: (..., n_steps, n_equations)
        iv_rhs: (..., n_init_var_steps, n_dims, n_init_var_orders)
        steps: (..., n_steps-1, n_dims)
        """

        rhs = rhs.reshape(self.batch_size, self.n_steps)
        iv_rhs = iv_rhs.reshape(self.batch_size, self.n_init_var_steps * self.n_init_orders)

        steps = steps.reshape(self.batch_size, self.n_steps-1, self.n_dims)

        # coeffs = torch.ones_like(coeffs)

        derivative_constraints = self.build_derivative_tensor(steps)
        eq_A = torch.sparse_coo_tensor(self.equation_indices, coeffs.flatten(), dtype=coeffs.dtype, device=coeffs.device)
        A = torch.cat([eq_A, self.initial_A.to(dtype=coeffs.dtype, device=coeffs.device), derivative_constraints], dim=-2).to_dense()
        beta = torch.cat([rhs, iv_rhs], dim=1).to(dtype=A.dtype)

        A = A[..., 1:]
        At = A.transpose(-2, -1)
        AtA = At @ A
        L, info = torch.linalg.cholesky_ex(AtA, upper=False, check_errors=False)

        # from PIL import Image
        #
        # def save_mat(matrix, fname):
        #     matrix = matrix[0].bool().detach().cpu().to(dtype=torch.uint8).numpy() * 255
        #     image = Image.fromarray(matrix)  # 'L' mode for grayscale
        #     image.save(f'logs/img/1p/{fname}.png')
        #
        # save_mat(A, 'a')
        # save_mat(AtA, 'ata')
        # save_mat(L, 'l')

        rhs1 = At[..., :beta.size(-1)] @ beta[..., None]
        x = rhs1.cholesky_solve(L, upper=False)[..., 0]

        # shape: batch, step, vars (== 1), order
        u = x.reshape(self.batch_size, self.n_steps, self.n_dims, self.n_orders)
        return u

    def build_derivative_tensor(self, steps: torch.Tensor):
        sign = 1

        # build forward values
        forward_list = []
        backward_list = []
        ones = torch.ones_like(steps)
        for i in range(self.n_orders - 1):
            forward_list.append(ones)
            backward_list.append(ones)
            ss = -sign * steps ** i
            forward_list.append(ss)
            backward_list.append(ss if i % 2 == 0 else -ss)
            for j in range(i, self.n_orders):
                ss = sign * steps ** j / math.factorial(j - i)
                forward_list.append(ss)
                backward_list.append(ss if j % 2 == 0 else -ss)
        fv = torch.stack(forward_list, dim=-1).flatten(start_dim=1)
        bv = torch.stack(backward_list, dim=-1).flatten(start_dim=1)

        # build central values
        # steps shape b, n_step-2, n_system_vars
        csteps = steps[:, 1:, :]
        psteps = steps[:, :-1, :]
        ones = torch.ones_like(csteps)
        # scale to make error of order O(h^3) for second order O(h^2) for first order
        cpsteps = csteps + psteps
        mult = cpsteps ** (self.n_orders - 2)
        sum_inv = cpsteps ** (self.n_orders - 3)
        # shape: b, n_steps-1, 4
        values = torch.stack([ones, -sum_inv, sum_inv, -mult], dim=-1)
        # shape, b, n_step-1, n_system_vars, n_order-1, 4
        cv = values.flatten(start_dim=1)

        derivative_values = torch.cat([fv, cv, bv], dim=-1).flatten()
        return torch.sparse_coo_tensor(
            indices=self.smooth_indices, values=derivative_values, dtype=steps.dtype, device=steps.device,
        )
