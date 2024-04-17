from enum import Enum, IntEnum
import itertools
import math
from typing import Dict, List

import numpy as np
import torch


class VarType(Enum):
    EPS = 1


class ConstraintType(Enum):
    Equation = 1
    Initial = 10
    Derivative = 20


PH = torch.nan  # placeholder


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

        # initial constraint steps starting from step 0
        self.num_constraints: int = 0

        # tracks number of added constraints
        self.num_added_constraints: int = 0
        self.num_added_equation_constraints: int = 0
        self.num_added_initial_constraints: int = 0
        self.num_added_derivative_constraints: int = 0

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
        # constraint coefficients
        self.value_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}
        # constraint indices
        self.row_dict: Dict[ConstraintType, List[int]] = {ConstraintType.Equation: [], ConstraintType.Initial: [],
                                                          ConstraintType.Derivative: []}
        # variable indices
        self.col_dict: Dict[ConstraintType, List[int]] = {ConstraintType.Equation: [], ConstraintType.Initial: [],
                                                          ConstraintType.Derivative: []}
        # rhs values
        self.rhs_dict = {ConstraintType.Equation: [], ConstraintType.Initial: [], ConstraintType.Derivative: []}

        # build skeleton constraints. filled during training
        # one equation for each dimension
        for _ in range(n_equations):
            for step in range(self.n_step):
                var_list = list(itertools.product([step], range(self.n_system_vars), range(self.n_order)))
                val_list = [PH] * len(var_list)
                self._add_constraint(var_list=var_list, values=val_list, rhs=PH,
                                     constraint_type=ConstraintType.Equation)

        self._build_derivative_constraints()

        # self._build_initial_constraints()
        # equation coefficients over grid
        for step, dim, i in itertools.product(range(n_iv_steps), range(n_dim), range(n_iv)):
            self._add_constraint(var_list=[(step, dim, i)], values=[1], rhs=PH,
                                 constraint_type=ConstraintType.Initial)
        if periodic_boundary:
            for dim, order in itertools.product(range(n_dim), range(self.n_order - 1)):
                self._add_constraint(var_list=[(0, dim, order), (self.n_step - 1, dim, order)], values=[1, -1],
                                     rhs=PH, constraint_type=ConstraintType.Initial)

        eq_A = torch.sparse_coo_tensor(
            indices=torch.tensor([self.row_dict[ConstraintType.Equation], self.col_dict[ConstraintType.Equation]]),
            values=torch.tensor(self.value_dict[ConstraintType.Equation]),
            size=[self.num_added_equation_constraints, self.num_vars],
            dtype=self.dtype,
            device=self.device
        )
        self.eq_A = torch.stack([eq_A] * self.bs, dim=0)  # (b, r1, c)

        derivative_A = torch.sparse_coo_tensor(
            indices=torch.tensor([self.row_dict[ConstraintType.Derivative], self.col_dict[ConstraintType.Derivative]]),
            values=self.value_dict[ConstraintType.Derivative],
            size=(self.num_added_derivative_constraints, self.num_vars),
            dtype=self.dtype,
            device=self.device,
        )
        self.derivative_A = torch.stack([derivative_A] * self.bs, dim=0)  # (b, r3, c)

        self.derivative_rhs = torch.tensor(self.rhs_dict[ConstraintType.Derivative], dtype=self.dtype,
                                           device=self.device).repeat(self.bs, 1)

        if n_iv > 0:
            initial_A = torch.sparse_coo_tensor(
                indices=torch.tensor([self.row_dict[ConstraintType.Initial], self.col_dict[ConstraintType.Initial]]),
                values=self.value_dict[ConstraintType.Initial],
                size=(self.num_added_initial_constraints, self.num_vars),
                dtype=self.dtype,
                device=self.device,
            )
            full_A = torch.cat([eq_A, initial_A, derivative_A], dim=0)
            self.initial_A = torch.stack([initial_A] * self.bs, dim=0)  # (b, r2, c)
        else:
            full_A = torch.cat([eq_A, derivative_A], dim=0)
            self.initial_A = None

        self.num_constraints: int = full_A.size(0)

    def get_solution_reshaped(self, x):
        """remove eps and reshape solution"""
        # x = x[:, 1:].reshape(-1, *self.multi_index_shape)
        x = x.reshape(-1, *self.multi_index_shape)
        return x

    def _add_constraint(self, var_list, values, rhs, constraint_type):
        """ var_list: list of multindex tuples or eps enum """
        if constraint_type == ConstraintType.Equation:
            constraint_index = self.num_added_equation_constraints
        elif constraint_type == ConstraintType.Initial:
            constraint_index = self.num_added_initial_constraints
        elif constraint_type == ConstraintType.Derivative:
            constraint_index = self.num_added_derivative_constraints

        for i, v in enumerate(var_list):
            if v == VarType.EPS:
                var_index = 0  # eps has index 0
            else:
                # 0 is epsilon, step, grad_index
                var_index = np.ravel_multi_index(v, self.multi_index_shape, order='C') + 1

            self.col_dict[constraint_type].append(var_index)
            self.value_dict[constraint_type].append(values[i])
            self.row_dict[constraint_type].append(constraint_index)

        self.rhs_dict[constraint_type].append(rhs)

        self.num_added_constraints = self.num_added_constraints + 1
        if constraint_type == ConstraintType.Equation:
            self.num_added_equation_constraints += 1
        elif constraint_type == ConstraintType.Initial:
            self.num_added_initial_constraints += 1
        elif constraint_type == ConstraintType.Derivative:
            self.num_added_derivative_constraints += 1

    def _build_derivative_constraints(self):
        sign = 1

        for step, dim, i in itertools.product(range(self.n_step - 1), range(self.n_system_vars),
                                              range(self.n_order - 1)):
            # forward constraints
            var_list = [VarType.EPS, (step + 1, dim, i)]
            val_list = [1., -sign * self.step_list[step] ** i]
            for j in range(i, self.n_order):
                var_list.append((step, dim, j))
                val_list.append(sign * self.step_list[step] ** j / math.factorial(j - i))
            self._add_constraint(var_list=var_list, values=val_list, rhs=0., constraint_type=ConstraintType.Derivative)

        # central constraints
        # central difference for derivatives
        for step, dim in itertools.product(range(1, self.n_step - 1), range(self.n_system_vars)):
            self._add_constraint(
                var_list=[
                    VarType.EPS,
                    (step - 1, dim, self.n_order - 2),
                    (step + 1, dim, self.n_order - 2),
                    (step, dim, self.n_order - 1)
                ],
                values=[-1., -.5 / self.step_size, .5 / self.step_size, -1.],
                rhs=0.,
                constraint_type=ConstraintType.Derivative,
            )

        for step, dim, i in itertools.product(range(self.n_step - 1), range(self.n_system_vars),
                                              range(self.n_order - 1)):
            # backward constraints
            var_list = [VarType.EPS, (step, dim, i)]
            val_list = [1., -sign * (-self.step_list[step]) ** i]
            for j in range(i, self.n_order):
                var_list.append((step + 1, dim, j))
                val_list.append(sign * (-self.step_list[step]) ** j / math.factorial(j - i))
            self._add_constraint(var_list=var_list, values=val_list, rhs=0, constraint_type=ConstraintType.Derivative)

    def _get_row_col_sorted_indices(self, row, col):
        """ Compute indices sorted by row and column and repeats. Useful for sparse outer product when computing constraint derivatives"""
        indices = torch.tensor(np.stack([row, col], axis=0))
        row_sorted = indices[:, indices[0, :].argsort()]
        column_sorted = indices[:, indices[1, :].argsort()]
        _, row_counts = row_sorted[0].unique_consecutive(return_counts=True)
        _, column_counts = column_sorted[1].unique_consecutive(return_counts=True)
        # add batch dimension
        batch_dim = torch.arange(self.bs).repeat_interleave(repeats=row.shape[0])[None]
        row_sorted = torch.cat([batch_dim, row_sorted.repeat(1, self.bs)], dim=0)
        column_sorted = torch.cat([batch_dim, column_sorted.repeat(1, self.bs)], dim=0)
        return row_sorted, column_sorted, row_counts, column_counts

    def build_equation_tensor(self, eq_values):
        # shape batch, n_eq, n_step, n_vars, order+1
        eq_values = eq_values.reshape(-1)
        eq_indices = self.eq_A._indices()
        G = torch.sparse_coo_tensor(eq_indices, eq_values, dtype=self.dtype, device=eq_values.device)
        return G

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
        G = torch.sparse_coo_tensor(indices=derivative_indices, values=derivative_values, dtype=self.dtype,
                                    device=steps.device)
        return G

    def build_ode(self, eq_A, eq_rhs, iv_rhs, derivative_A):
        if derivative_A is None:
            derivative_A = self.derivative_A

        if self.initial_A is not None:
            self.AG = torch.cat([eq_A, self.initial_A.type_as(derivative_A), derivative_A], dim=1)
        else:
            self.AG = torch.cat([eq_A, derivative_A], dim=1)

        self.num_constraints = self.AG.shape[1]

        self.derivative_rhs = self.derivative_rhs.type_as(eq_rhs)
        if self.initial_A is not None:
            self.ub = torch.cat([eq_rhs, iv_rhs, self.derivative_rhs], dim=1)
        else:
            self.ub = torch.cat([eq_rhs, self.derivative_rhs], dim=1)
