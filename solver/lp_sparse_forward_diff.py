from enum import Enum
import itertools
import math

import numpy as np
import torch


class VarType(Enum):
    EPS = 1


class ODESYSLP:
    def __init__(
            self,
            bs: int = 1,
            n_step: int = 3,
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
        self.n_step: int = n_step
        self.step_size: float = step_size
        self.step_list = torch.full([n_step - 1], step_size) if step_list is None else step_list

        # order is diffeq order. n_order is total number of terms: y'', y', y for order 2.
        self.n_order: int = order + 1
        # number of ode variables
        # number of auxiliary variables per dim for non-linear terms
        # dimensions plus n_auxliary vars for each dim
        self.n_system_vars: int = n_dim * (1 + n_auxiliary)
        # batch size
        self.bs: int = bs
        self.dtype = dtype
        self.device: torch.device = device

        # total number of qp variables
        self.num_vars = self.n_system_vars * self.n_step * self.n_order + 1
        # Variables except eps. Used for raveling
        self.multi_index_shape = (self.n_step, self.n_system_vars, self.n_order)

        #### sparse constraint arrays
        self.equations = []
        self.init_equations = []
        self.smooth_equations = []

        PH = torch.nan  # placeholder

        # build skeleton constraints. filled during training
        # one equation for each dimension
        self.equations += [{
            'lhs': {
                (step, dim, order): PH for dim, order in itertools.product(range(self.n_system_vars), range(self.n_order))
            },
            'rhs': PH,
            'label': 'eq',
        } for step in range(self.n_step)]

        sign = 1

        self.smooth_equations += [{
            'lhs': {
                VarType.EPS: 1.,
                (step + 1, dim, i): -sign * self.step_list[step] ** i,
                **{
                    (step, dim, j): sign * self.step_list[step] ** j / math.factorial(j - i)
                    for j in range(i, self.n_order)
                }
            },
            'rhs': 0.,
            'label': 'smf',
        } for step, dim, i in itertools.product(range(self.n_step - 1), range(self.n_system_vars),
                                              range(self.n_order - 1))]

        self.smooth_equations += [{
            'lhs': {
                VarType.EPS: -1.,
                (step - 1, dim, self.n_order - 2): -.5 / self.step_size,
                (step + 1, dim, self.n_order - 2): .5 / self.step_size,
                (step, dim, self.n_order - 1): -1.,
            },
            'rhs': 0.,
            'label': 'smc',
        } for step, dim in itertools.product(range(1, self.n_step - 1), range(self.n_system_vars))]

        self.smooth_equations += [{
            'lhs': {
                VarType.EPS: 1.,
                (step, dim, i): -sign * (-self.step_list[step]) ** i,
                **{
                    (step + 1, dim, j): sign * (-self.step_list[step]) ** j / math.factorial(j - i)
                    for j in range(i, self.n_order)
                }
            },
            'rhs': 0.,
            'label': 'smb',
        } for step, dim, i in itertools.product(range(self.n_step - 1), range(self.n_system_vars),
                                              range(self.n_order - 1))]

        self.init_equations += [{
            'lhs': {
                (step, dim, order): 1.,
            },
            'rhs': PH,
            'label': 'in',
        } for step, dim, order in itertools.product(range(n_iv_steps), range(n_dim), range(n_iv))]

        if periodic_boundary:
            self.init_equations += [{
                'lhs': {
                    (0, dim, order): 1.,
                    (self.n_step - 1, dim, order): -1.,
                },
                'rhs': PH,
                'label': 'in',
            } for dim, order in itertools.product(range(n_dim), range(self.n_order - 1))]

        eq_indices = []
        eq_values = []
        for i, equation in enumerate(self.equations):
            for k, v in equation['lhs'].items():
                eq_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1])
                eq_values.append(v)
        eq_size = len(self.equations)

        eq_A = torch.sparse_coo_tensor(
            indices=torch.tensor(eq_indices).t(),
            values=torch.tensor(eq_values),
            size=[eq_size, self.num_vars],
            dtype=self.dtype,
            device=self.device
        )
        self.eq_A = torch.stack([eq_A] * self.bs, dim=0)  # (b, r1, c)

        eq_indices = []
        eq_values = []
        for i, equation in enumerate(self.init_equations):
            for k, v in equation['lhs'].items():
                eq_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1])
                eq_values.append(v)
        eq_size = len(self.init_equations)

        initial_A = torch.sparse_coo_tensor(
            indices=torch.tensor(eq_indices).t(),
            values=torch.tensor(eq_values),
            size=[eq_size, self.num_vars],
            dtype=self.dtype,
            device=self.device
        )
        self.initial_A = torch.stack([initial_A] * self.bs, dim=0)  # (b, r1, c)

        eq_indices = []
        eq_values = []
        for i, equation in enumerate(self.smooth_equations):
            for k, v in equation['lhs'].items():
                eq_indices.append([i, np.ravel_multi_index(k, self.multi_index_shape, order='C') + 1] if k != VarType.EPS else [i, 0])
                eq_values.append(v)
        eq_size = len(self.smooth_equations)

        derivative_A = torch.sparse_coo_tensor(
            indices=torch.tensor(eq_indices).t(),
            values=torch.tensor(eq_values),
            size=[eq_size, self.num_vars],
            dtype=self.dtype,
            device=self.device
        )
        self.derivative_A = torch.stack([derivative_A] * self.bs, dim=0)  # (b, r1, c)

        self.derivative_rhs = torch.tensor([equation['rhs'] for equation in self.smooth_equations], dtype=self.dtype,
                                           device=self.device).repeat(self.bs, 1)

    def get_solution_reshaped(self, x):
        """remove eps and reshape solution"""
        # x = x[:, 1:].reshape(-1, *self.multi_index_shape)
        x = x.reshape(-1, *self.multi_index_shape)
        return x

    def build_derivative_tensor(self, steps: torch.Tensor):
        sign = 1

        # build forward values
        order_list = []
        for i in range(self.n_order - 1):
            order_list.append(torch.ones_like(steps))
            order_list.append(-sign * steps ** i)
            for j in range(i, self.n_order):
                order_list.append(sign * steps ** j / math.factorial(j - i))
        fv = torch.stack(order_list, dim=-1).flatten(start_dim=1)

        # build central values
        # steps shape b,  n_step-1, n_system_vars,
        csteps = steps[:, 1:, :]
        psteps = steps[:, :-1, :]
        sum_inv = 1. / (csteps + psteps)
        ones = torch.ones_like(csteps)
        # scale to make error of order O(h^3) for second order O(h^2) for first order
        mult = (csteps + psteps) ** (self.n_order - 2)
        # shape: b, n_steps-1, 4
        values = torch.stack([ones, -sum_inv * mult, sum_inv * mult, -mult], dim=-1)
        # flatten
        # shape, b, n_step-1, n_system_vars, n_order-1, 4
        cv = values.flatten(start_dim=1)

        # build backward values
        # no reversing
        order_list = []
        for i in range(self.n_order - 1):
            order_list.append(torch.ones_like(steps))
            order_list.append(-sign * (-steps) ** i)
            for j in range(i, self.n_order):
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

        self.derivative_rhs = self.derivative_rhs.type_as(eq_rhs)
        self.ub = torch.cat([eq_rhs, iv_rhs, self.derivative_rhs], dim=1)
