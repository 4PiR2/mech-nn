import torch
from torch import nn


class ODEINDLayer(nn.Module):
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

    def forward(
            self,
            coefficients: torch.Tensor,
            rhs_equation: torch.Tensor,
            rhs_init: torch.Tensor,
            steps: torch.Tensor,
    ) -> torch.Tensor:
        """
        coefficients: (..., n_steps, n_equations, n_dims, n_orders)
        rhs_equation: (..., n_steps, n_equations)
        rhs_init: (..., n_init_var_steps, n_dims, n_init_var_orders)
        steps: (..., n_steps-1)
        return: (..., n_steps, n_dims, n_orders)
        """

        dtype = coefficients.dtype
        device: torch.device = coefficients.device

        *batches, n_steps, n_equations, n_dims, n_orders = coefficients.shape
        *_, n_init_var_steps, _, n_init_var_orders = rhs_init.shape
        assert n_steps, n_equations == rhs_equation.shape[-2:]
        assert n_dims == rhs_init.shape[-2]
        assert n_steps - 1 == steps.shape[-1]

        # ode equation constraints
        c = coefficients.flatten(start_dim=-2)  # (..., n_steps, n_equations, n_dims * n_orders)
        ct = c.transpose(-2, -1)  # (..., n_steps, n_dims * n_orders, n_equations)
        block_diag_0 = ct @ c  # (..., n_steps, n_dims * n_orders, n_dims * n_orders)
        beta = (ct @ rhs_equation[..., None])  # (..., n_steps, n_dims * n_orders, 1)

        # initial-value constraints
        init_idx = torch.arange(n_init_var_orders, device=device).repeat(n_dims) \
                   + n_orders * torch.arange(n_dims, device=device).repeat_interleave(n_init_var_orders)
                   # (n_dims * n_init_var_orders)
        block_diag_0[..., :n_init_var_steps, init_idx, init_idx] += 1.
        beta[..., :n_init_var_steps, :, 0] += torch.cat([
            rhs_init,
            torch.zeros(*rhs_init.shape[:-1], n_orders - n_init_var_orders, dtype=dtype, device=device),
        ], dim=-1).flatten(start_dim=-2)

        # smoothness constraints (forward & backward)
        order_idx = torch.arange(n_orders, device=device)  # (n_orders)
        sign_vec = order_idx % 2 * (-2) + 1  # (n_orders)
        sign_map = sign_vec * sign_vec[:, None]  # (n_orders, n_orders)

        expansions = steps[..., None] ** order_idx  # (..., n_steps-1, n_orders)
        et_e_diag = expansions ** 2  # (..., n_steps-1, n_orders)
        et_e_diag[..., -1] = 0.
        factorials = (-(order_idx - order_idx[:, None] + 1).triu().lgamma()).exp()  # (n_orders, n_orders)
        factorials[-1, -1] = 0.
        e_outer = expansions[..., None] * expansions[..., None, :]  # (..., n_steps-1, n_orders, n_orders)
        et_ft_f_e = e_outer * (factorials.t() @ factorials)  # (..., n_steps-1, n_orders, n_orders)

        smooth_block_diag_1 = e_outer * -(factorials + factorials.transpose(-2, -1) * sign_map)  # (..., n_steps-1, n_orders, n_orders)
        smooth_block_diag_0 = torch.zeros(*batches, n_steps, n_orders, n_orders, dtype=dtype, device=device)  # (..., n_steps, n_orders, n_orders)
        smooth_block_diag_0[..., :-1, :, :] += et_ft_f_e
        smooth_block_diag_0[..., 1:, :, :] += et_ft_f_e * sign_map
        smooth_block_diag_0[..., :-1, order_idx, order_idx] += et_e_diag
        smooth_block_diag_0[..., 1:, order_idx, order_idx] += et_e_diag

        # smoothness constraints (central)
        steps2 = steps[..., :-1] + steps[..., 1:]  # (..., n_steps-2)
        steps26 = steps2 ** (n_orders * 2 - 6)  # (..., n_steps-2)
        steps25 = steps2 ** (n_orders * 2 - 5)  # (..., n_steps-2)
        steps24 = steps2 ** (n_orders * 2 - 4)  # (..., n_steps-2)

        smooth_block_diag_0[..., :-2, n_orders - 2, n_orders - 2] += steps26
        smooth_block_diag_0[..., 2:, n_orders - 2, n_orders - 2] += steps26
        smooth_block_diag_0[..., 1:-1, n_orders - 1, n_orders - 1] += steps24
        smooth_block_diag_1[..., :-1, n_orders - 1, n_orders - 2] += steps25
        smooth_block_diag_1[..., 1:, n_orders - 2, n_orders - 1] -= steps25
        smooth_block_diag_2 = torch.zeros(*batches, n_steps - 2, n_orders, n_orders, dtype=dtype, device=device)  # (..., n_steps-2, n_orders, n_orders)
        smooth_block_diag_2[..., n_orders - 2, n_orders - 2] = -steps26

        # copy to n_dims
        block_diag_1 = torch.zeros(*batches, n_steps - 1, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device)  # (..., n_steps-1, n_dims * n_orders, n_dims * n_orders)
        block_diag_2 = torch.zeros(*batches, n_steps - 2, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device)  # (..., n_steps-2, n_dims * n_orders, n_dims * n_orders)
        for dim in range(n_dims):
            i1 = dim * n_orders
            i2 = (dim + 1) * n_orders
            block_diag_0[..., i1:i2, i1:i2] += smooth_block_diag_0
            block_diag_1[..., i1:i2, i1:i2] = smooth_block_diag_1
            block_diag_2[..., i1:i2, i1:i2] = smooth_block_diag_2

        # in-place blocked cholesky decomposition
        for step in range(n_steps):
            if step >= 2:
                torch.linalg.solve_triangular(
                    block_diag_0[..., step - 2, :, :].transpose(-2, -1),
                    block_diag_2[..., step - 2, :, :],
                    upper=True,
                    left=False,
                    out=block_diag_2[..., step - 2, :, :],
                )
                block_diag_1[..., step - 1, :, :] -= block_diag_2[..., step - 2, :, :] @ block_diag_1[..., step - 2, :, :].transpose(-2, -1)
            if step >= 1:
                torch.linalg.solve_triangular(
                    block_diag_0[..., step - 1, :, :].transpose(-2, -1),
                    block_diag_1[..., step - 1, :, :],
                    upper=True,
                    left=False,
                    out=block_diag_1[..., step - 1, :, :],
                )
                if step >= 2:
                    block_diag_0[..., step, :, :] -= block_diag_2[..., step - 2, :, :] @ block_diag_2[..., step - 2, :, :].transpose(-2, -1)
                block_diag_0[..., step, :, :] -= block_diag_1[..., step - 1, :, :] @ block_diag_1[..., step - 1, :, :].transpose(-2, -1)
            torch.linalg.cholesky_ex(
                block_diag_0[..., step, :, :],
                upper=False,
                check_errors=False,
                out=(block_diag_0[..., step, :, :], torch.empty(*batches, dtype=torch.int, device=device)),
            )

        # L = torch.zeros(*batches, n_steps * n_dims * n_orders, n_steps * n_dims * n_orders, dtype=dtype, device=device)
        # for step in range(n_steps):
        #     L[..., step * n_dims * n_orders: (step+1) * n_dims * n_orders, step * n_dims * n_orders: (step+1) * n_dims * n_orders] = block_diag_0[..., step, :, :]
        # for step in range(n_steps-1):
        #     L[..., (step+1) * n_orders: (step+2) * n_orders, step * n_orders: (step+1) * n_orders] = block_diag_1[..., step, :, :]
        # for step in range(n_steps-2):
        #     L[..., (step+2) * n_orders: (step+3) * n_orders, step * n_orders: (step+1) * n_orders] = block_diag_2[..., step, :, :]

        # x0 = beta.flatten(start_dim=-3, end_dim=-2).cholesky_solve(L, upper=False)  # (..., n_steps * n_dims * n_orders, 1)
        # y = torch.linalg.solve_triangular(L, beta.flatten(start_dim=-3, end_dim=-2), upper=False, left=True)
        # x0 = torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True, left=True)

        # A X = B => L (Lt X) = B
        # solve L Y = B, in-place block forward substitution
        for step in range(n_steps):
            beta[..., step, :, :] = torch.linalg.solve_triangular(block_diag_0[..., step, :, :], beta[..., step, :, :], upper=False, left=True)
            # function with "out=" does not support auto-diff
            if step < n_steps - 1:
                beta[..., step + 1, :, :] -= block_diag_1[..., step, :, :] @ beta[..., step, :, :]
            if step < n_steps - 2:
                beta[..., step + 2, :, :] -= block_diag_2[..., step, :, :] @ beta[..., step, :, :]

        block_diag_0 = block_diag_0.transpose(-2, -1)
        block_diag_1 = block_diag_1.transpose(-2, -1)
        block_diag_2 = block_diag_2.transpose(-2, -1)

        # solve Lt X = Y, in-place block backward substitution
        for step in range(n_steps - 1, -1, -1):
            beta[..., step, :, :] = torch.linalg.solve_triangular(block_diag_0[..., step, :, :], beta[..., step, :, :], upper=True, left=True)
            if step >= 1:
                beta[..., step - 1, :, :] -= block_diag_1[..., step - 1, :, :] @ beta[..., step, :, :]
            if step >= 2:
                beta[..., step - 2, :, :] -= block_diag_2[..., step - 2, :, :] @ beta[..., step, :, :]

        x = beta.reshape(*batches, n_steps, n_dims, n_orders)
        return x
