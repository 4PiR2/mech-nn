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
        steps: (..., n_steps-1)
        """

        steps = steps.reshape(self.batch_size, self.n_steps-1)

        # coeffs = torch.ones_like(coeffs)

        derivative_constraints = self.build_derivative_tensor(steps.reshape(self.batch_size, self.n_steps-1, 1))

        Ae = coeffs.flatten(start_dim=-2)  # (..., n_steps, n_equations, n_dims * n_orders)
        Aet = Ae.transpose(-2, -1)  # (..., n_steps, n_dims * n_orders, n_equations)
        Aet_Ae = Aet @ Ae  # (..., n_steps, n_dims * n_orders, n_dims * n_orders)
        Aet_be = (Aet @ rhs[..., None])[..., 0]  # (..., n_steps, n_dims * n_orders)

        init_idx = torch.arange(self.n_init_orders).repeat(self.n_dims) + torch.arange(self.n_dims).repeat_interleave(self.n_init_orders) * self.n_orders
        Aet_Ae[..., :self.n_init_var_steps, init_idx, init_idx] += 1.
        bi = torch.cat([iv_rhs, torch.zeros(*iv_rhs.shape[:-1], self.n_orders - iv_rhs.size(-1), dtype=iv_rhs.dtype, device=iv_rhs.device)], dim=-1)
        Aet_be[..., :self.n_init_var_steps, :] += bi.flatten(start_dim=-2)

        order_idx = torch.arange(self.n_orders, device=steps.device)  # (n_orders)
        sign_vec = (-1) ** order_idx  # (n_orders)
        sign_map = sign_vec * sign_vec[:, None]  # (n_orders, n_orders)

        expansions = steps[..., None] ** order_idx  # (..., n_steps-1, n_orders)
        et_e_diag = expansions ** 2  # (..., n_steps-1, n_orders)
        et_e_diag[..., -1] = 0.

        factorials = (-(order_idx - order_idx[:, None] + 1).triu().lgamma()).exp()  # (n_orders, n_orders)
        factorials[-1, -1] = 0.
        e_outer = expansions[..., None] * expansions[..., None, :]  # (..., n_steps-1, n_orders, n_orders)
        et_f_e = factorials * e_outer  # (..., n_steps-1, n_orders, n_orders)
        et_ft_f_e = (factorials.t() @ factorials) * e_outer  # (..., n_steps-1, n_orders, n_orders)

        block_lower_off_diag = - et_f_e - (et_f_e * sign_map).transpose(-2, -1)  # (..., n_steps-1, n_orders, n_orders)
        block_diag = torch.zeros(*block_lower_off_diag.shape[:-3], self.n_steps, self.n_orders, self.n_orders, dtype=steps.dtype, device=steps.device)
        block_diag[..., :-1, :, :] += et_ft_f_e
        block_diag[..., 1:, :, :] += et_ft_f_e * sign_map
        block_diag[..., :-1, order_idx, order_idx] += et_e_diag
        block_diag[..., 1:, order_idx, order_idx] += et_e_diag

        Aet_Ae_dense = torch.zeros(*Aet_Ae.shape[:-3], self.n_steps * self.n_dims * self.n_orders, self.n_steps * self.n_dims * self.n_orders, dtype=Aet_Ae.dtype, device=Aet_Ae.device)
        for step in range(self.n_steps):
            Aet_Ae_dense[..., step * self.n_dims * self.n_orders: (step+1) * self.n_dims * self.n_orders, step * self.n_dims * self.n_orders: (step+1) * self.n_dims * self.n_orders] = Aet_Ae[..., step, :, :]
        Aet_be_dense = Aet_be.flatten(start_dim=-2)  # (..., n_steps * n_dims * n_orders)

        Ast_As_dense = torch.zeros_like(Aet_Ae_dense)
        for step in range(self.n_steps):
            Ast_As_dense[..., step * self.n_orders: (step+1) * self.n_orders, step * self.n_orders: (step+1) * self.n_orders] = block_diag[..., step, :, :]
        for step in range(self.n_steps-1):
            Ast_As_dense[..., (step+1) * self.n_orders: (step+2) * self.n_orders, step * self.n_orders: (step+1) * self.n_orders] = block_lower_off_diag[..., step, :, :]
            Ast_As_dense[..., step * self.n_orders: (step+1) * self.n_orders, (step+1) * self.n_orders: (step+2) * self.n_orders] = block_lower_off_diag[..., step, :, :].transpose(-2, -1)

        A = derivative_constraints.to_dense()[..., 1:]
        AtA = A.transpose(-2, -1) @ A + Aet_Ae_dense
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

        x = Aet_be_dense[..., None].cholesky_solve(L, upper=False)[..., 0]

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
