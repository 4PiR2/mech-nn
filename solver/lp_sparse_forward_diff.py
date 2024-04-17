import itertools
import math

import numpy as np
import torch


class ODESYSLP:
    def __init__(
            self,
            batch_size: int = 1,
            n_steps: int = 3,
            n_dims: int = 1,
            ode_order: int = 2,
            n_init_var_steps: int = 1,
            n_init_orders: int = 2,
            periodic_boundary: bool = False,
            dtype=torch.float64,
            device: torch.device = None,
    ):
        super().__init__()

        # order is diffeq order. n_order is total number of terms: y'', y', y for order 2.
        self.n_orders: int = ode_order + 1

        # total number of qp variables
        n_vars = n_dims * n_steps * self.n_orders + 1

        # Variables except eps. Used for raveling
        self.multi_index_shape = (n_steps, n_dims, self.n_orders)

        # build skeleton constraints, one equation for each dimension, filled during training
        equation_constraints = [
            [(step, dim, order) for dim, order in itertools.product(range(n_dims), range(self.n_orders))]
            for step in range(n_steps)
        ]
        constraint_indices = []
        for i, equation in enumerate(equation_constraints):
            for k in equation:
                constraint_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1])
        self.equation_indices = torch.cat([
            torch.arange(batch_size).repeat_interleave(len(constraint_indices))[None],
            torch.cat([torch.tensor(constraint_indices).t()] * batch_size, dim=-1),
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
            torch.cat([torch.tensor(constraint_indices).t()] * batch_size, dim=-1),
        ], dim=-2)

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
        bv = torch.stack(backward_list, dim=-1).flatten(start_dim=1)  # b, n_steps-1, n_system_vars, n_order+2

        # build central values
        # steps shape b, n_step-1, n_system_vars,
        csteps = steps[:, 1:, :]
        psteps = steps[:, :-1, :]
        ones = torch.ones_like(csteps)
        # scale to make error of order O(h^3) for second order O(h^2) for first order
        cpsteps = csteps + psteps
        mult = cpsteps ** (self.n_orders - 2)
        sum_inv = cpsteps ** (self.n_orders - 3)
        # shape: b, n_steps-1, 4
        values = torch.stack([ones, -sum_inv, sum_inv, -mult], dim=-1)
        # flatten
        # shape, b, n_step-1, n_system_vars, n_order-1, 4
        cv = values.flatten(start_dim=1)

        derivative_values = torch.cat([fv, cv, bv], dim=-1).flatten()
        return torch.sparse_coo_tensor(
            indices=self.smooth_indices, values=derivative_values, dtype=steps.dtype, device=steps.device,
        )

    def build_ode(self, coeffs, derivative_A):
        # shape batch, n_eq, n_step, n_vars, order+1
        eq_A = torch.sparse_coo_tensor(self.equation_indices, coeffs.flatten(), dtype=coeffs.dtype, device=coeffs.device)
        return torch.cat([eq_A, self.initial_A.to(dtype=coeffs.dtype, device=coeffs.device), derivative_A], dim=-2)
