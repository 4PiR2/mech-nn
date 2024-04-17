import itertools
import math

import numpy as np
import torch


EPS = 1


class ODESYSLP:
    def __init__(
            self,
            bs: int = 1,
            n_steps: int = 3,
            n_dim: int = 1,
            n_iv: int = 2,
            n_auxiliary: int = 0,
            n_equations: int = 1,
            step_size: float = .25,
            order: int = 2,
            periodic_boundary: bool = False,
            dtype=torch.float64,
            n_iv_steps: int = 1,
            step_list=None,
            device=None,
    ):
        super().__init__()
        n_steps: int = n_steps
        self.step_size: float = step_size
        self.step_list = torch.full([n_steps - 1], step_size) if step_list is None else step_list

        # order is diffeq order. n_order is total number of terms: y'', y', y for order 2.
        self.n_orders: int = order + 1
        # number of ode variables
        # number of auxiliary variables per dim for non-linear terms
        # dimensions plus n_auxliary vars for each dim
        n_system_vars: int = n_dim * (1 + n_auxiliary)
        # batch size
        batch_size: int = bs
        self.dtype = dtype
        device: torch.device = device

        # total number of qp variables
        num_vars = n_system_vars * n_steps * self.n_orders + 1
        # Variables except eps. Used for raveling
        self.multi_index_shape = (n_steps, n_system_vars, self.n_orders)

        #### sparse constraint arrays
        self.equation_constraints = []
        self.initial_constraints = []
        self.smooth_constraints = []

        # build skeleton constraints. filled during training
        # one equation for each dimension
        self.equation_constraints += [{
            (step, dim, order): None for dim, order in itertools.product(range(n_system_vars), range(self.n_orders))
        } for step in range(n_steps)]

        constraint_indices = []
        for i, equation in enumerate(self.equation_constraints):
            for k, v in equation.items():
                constraint_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1])

        eq_A = torch.sparse_coo_tensor(
            indices=torch.tensor(constraint_indices).t(),
            values=torch.empty(len(constraint_indices)),
            size=[len(self.equation_constraints), num_vars],
            dtype=self.dtype,
            device=device
        )
        self.eq_A = torch.stack([eq_A] * batch_size, dim=0)  # (b, r1, c)

        self.initial_constraints += [{
            (step, dim, order): 1.,
        } for step, dim, order in itertools.product(range(n_iv_steps), range(n_dim), range(n_iv))]

        if periodic_boundary:
            self.initial_constraints += [{
                (0, dim, order): 1.,
                (n_steps - 1, dim, order): -1.,
            } for dim, order in itertools.product(range(n_dim), range(self.n_orders - 1))]

        constraint_indices = []
        constraint_values = []
        for i, equation in enumerate(self.initial_constraints):
            for k, v in equation.items():
                constraint_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1])
                constraint_values.append(v)

        initial_A = torch.sparse_coo_tensor(
            indices=torch.tensor(constraint_indices).t(),
            values=torch.tensor(constraint_values),
            size=[len(self.initial_constraints), num_vars],
            dtype=self.dtype,
            device=device
        )
        self.initial_A = torch.stack([initial_A] * batch_size, dim=0)  # (b, r1, c)

        # forward
        self.smooth_constraints += [{
            EPS: None,
            (step + 1, dim, i): None,
            **{
                (step, dim, j): None for j in range(i, self.n_orders)
            }
        } for step, dim, i in itertools.product(range(n_steps - 1), range(n_system_vars),
                                                range(self.n_orders - 1))]
        # central
        self.smooth_constraints += [{
            EPS: None,
            (step - 1, dim, self.n_orders - 2): None,
            (step + 1, dim, self.n_orders - 2): None,
            (step, dim, self.n_orders - 1): None,
        } for step, dim in itertools.product(range(1, n_steps - 1), range(n_system_vars))]

        # backward
        self.smooth_constraints += [{
            EPS: None,
            (step, dim, i): None,
            **{
                (step + 1, dim, j): None for j in range(i, self.n_orders)
            }
        } for step, dim, i in itertools.product(range(n_steps - 1), range(n_system_vars),
                                                range(self.n_orders - 1))]

        constraint_indices = []
        for i, equation in enumerate(self.smooth_constraints):
            for k, v in equation.items():
                constraint_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1] if k != EPS else [i, 0])

        derivative_A = torch.sparse_coo_tensor(
            indices=torch.tensor(constraint_indices).t(),
            values=torch.empty(len(constraint_indices)),
            size=[len(self.smooth_constraints), num_vars],
            dtype=self.dtype,
            device=device
        )
        self.derivative_A = torch.stack([derivative_A] * batch_size, dim=0)  # (b, r1, c)

    def get_solution_reshaped(self, x):
        """remove eps and reshape solution"""
        # x = x[:, 1:].reshape(-1, *self.multi_index_shape)
        x = x.reshape(-1, *self.multi_index_shape)
        return x

    def build_derivative_tensor(self, steps: torch.Tensor):
        sign = 1

        # build forward values
        order_list = []
        for i in range(self.n_orders - 1):
            order_list.append(torch.ones_like(steps))
            order_list.append(-sign * steps ** i)
            for j in range(i, self.n_orders):
                order_list.append(sign * steps ** j / math.factorial(j - i))
        fv = torch.stack(order_list, dim=-1).flatten(start_dim=1)

        # build central values
        # steps shape b,  n_step-1, n_system_vars,
        csteps = steps[:, 1:, :]
        psteps = steps[:, :-1, :]
        sum_inv = 1. / (csteps + psteps)
        ones = torch.ones_like(csteps)
        # scale to make error of order O(h^3) for second order O(h^2) for first order
        mult = (csteps + psteps) ** (self.n_orders - 2)
        # shape: b, n_steps-1, 4
        values = torch.stack([ones, -sum_inv * mult, sum_inv * mult, -mult], dim=-1)
        # flatten
        # shape, b, n_step-1, n_system_vars, n_order-1, 4
        cv = values.flatten(start_dim=1)

        # build backward values
        # no reversing
        order_list = []
        for i in range(self.n_orders - 1):
            order_list.append(torch.ones_like(steps))
            order_list.append(-sign * (-steps) ** i)
            for j in range(i, self.n_orders):
                order_list.append(sign * (-steps) ** j / math.factorial(j - i))
        bv = torch.stack(order_list, dim=-1).flatten(start_dim=1)  # b, n_steps-1, n_system_vars, n_order+2

        derivative_values = torch.cat([fv, cv, bv], dim=-1).flatten()
        derivative_indices = self.derivative_A._indices()
        derivative_A = torch.sparse_coo_tensor(indices=derivative_indices, values=derivative_values, dtype=self.dtype,
                                    device=steps.device)
        return derivative_A

    def build_ode(self, coeffs, eq_rhs, iv_rhs, derivative_A):
        # shape batch, n_eq, n_step, n_vars, order+1
        eq_values = coeffs.reshape(-1)
        eq_indices = self.eq_A._indices()
        eq_A = torch.sparse_coo_tensor(eq_indices, eq_values, dtype=self.dtype, device=eq_values.device)

        self.AG = torch.cat([eq_A, self.initial_A.type_as(derivative_A), derivative_A], dim=1)
        self.ub = torch.cat([eq_rhs, iv_rhs], dim=1)
