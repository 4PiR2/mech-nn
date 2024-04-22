import torch


def ode_forward(
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

    smooth_block_diag_1 = e_outer * -(factorials + factorials.transpose(-2, -1) * sign_map)
    # (..., n_steps-1, n_orders, n_orders)
    smooth_block_diag_0 = torch.zeros(*batches, n_steps, n_orders, n_orders, dtype=dtype, device=device)
    # (..., n_steps, n_orders, n_orders)
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
    smooth_block_diag_2 = torch.zeros(*batches, n_steps - 2, n_orders, n_orders, dtype=dtype, device=device)
    # (..., n_steps-2, n_orders, n_orders)
    smooth_block_diag_2[..., n_orders - 2, n_orders - 2] = -steps26

    # copy to n_dims
    block_diag_1 = torch.zeros(*batches, n_steps - 1, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device)
    # (..., n_steps-1, n_dims * n_orders, n_dims * n_orders)
    block_diag_2 = torch.zeros(*batches, n_steps - 2, n_dims * n_orders, n_dims * n_orders, dtype=dtype, device=device)
    # (..., n_steps-2, n_dims * n_orders, n_dims * n_orders)
    for dim in range(n_dims):
        i1 = dim * n_orders
        i2 = (dim + 1) * n_orders
        block_diag_0[..., i1:i2, i1:i2] += smooth_block_diag_0
        block_diag_1[..., i1:i2, i1:i2] = smooth_block_diag_1
        block_diag_2[..., i1:i2, i1:i2] = smooth_block_diag_2

    # blocked cholesky decomposition
    block_diag_0_list: list[torch.Tensor] = list(block_diag_0.unbind(dim=-3))
    block_diag_1_list: list[torch.Tensor] = list(block_diag_1.unbind(dim=-3))
    block_diag_2_list: list[torch.Tensor] = list(block_diag_2.unbind(dim=-3))
    for step in range(n_steps):
        if step >= 2:
            block_diag_2_list[step - 2] = torch.linalg.solve_triangular(
                block_diag_0_list[step - 2].transpose(-2, -1),
                block_diag_2_list[step - 2],
                upper=True,
                left=False,
            )
            block_diag_1_list[step - 1] = block_diag_1_list[step - 1] \
                                          - block_diag_2_list[step - 2] @ block_diag_1_list[step - 2].transpose(-2, -1)
        if step >= 1:
            block_diag_1_list[step - 1] = torch.linalg.solve_triangular(
                block_diag_0_list[step - 1].transpose(-2, -1),
                block_diag_1_list[step - 1],
                upper=True,
                left=False,
            )
            if step >= 2:
                block_diag_0_list[step] = block_diag_0_list[step] \
                                          - block_diag_2_list[step - 2] @ block_diag_2_list[step - 2].transpose(-2, -1)
            block_diag_0_list[step] = block_diag_0_list[step] \
                                      - block_diag_1_list[step - 1] @ block_diag_1_list[step - 1].transpose(-2, -1)
        block_diag_0_list[step], _ = torch.linalg.cholesky_ex(
            block_diag_0_list[step],
            upper=False,
            check_errors=False,
        )

    # A X = B => L (Lt X) = B
    # solve L Y = B, block forward substitution
    b_list: list[torch.Tensor] = list(beta.unbind(dim=-3))
    y_list: list[torch.Tensor | None] = [None] * n_steps
    for step in range(n_steps):
        b_step = b_list[step]
        if step >= 2:
            b_step = b_step - block_diag_2_list[step - 2] @ y_list[step - 2]
        if step >= 1:
            b_step = b_step - block_diag_1_list[step - 1] @ y_list[step - 1]
        y_list[step] = torch.linalg.solve_triangular(
            block_diag_0_list[step],
            b_step,
            upper=False,
            left=True,
        )

    # solve Lt X = Y, block backward substitution
    x_list: list[torch.Tensor | None] = [None] * n_steps
    for step in range(n_steps - 1, -1, -1):
        y_step = y_list[step]
        if step < n_steps - 2:
            y_step = y_step - block_diag_2_list[step].transpose(-2, -1) @ x_list[step + 2]
        if step < n_steps - 1:
            y_step = y_step - block_diag_1_list[step].transpose(-2, -1) @ x_list[step + 1]
        x_list[step] = torch.linalg.solve_triangular(
            block_diag_0_list[step].transpose(-2, -1),
            y_step,
            upper=True,
            left=True,
        )

    u = torch.stack(x_list, dim=-3).reshape(*batches, n_steps, n_dims, n_orders)
    return u


def test():
    torch.autograd.set_detect_anomaly(mode=True, check_nan=True)
    dtype = torch.float64
    device = torch.device('cuda:0')
    batches = (11,)
    n_steps, n_equations, n_dims, n_orders = 7, 2, 3, 5
    n_init_var_steps, n_init_var_orders = 3, 4
    coefficients = torch.nn.Parameter(torch.randn(*batches, n_steps, n_equations, n_dims, n_orders, dtype=dtype, device=device))
    rhs_equation = torch.nn.Parameter(torch.randn(*batches, n_steps, n_equations, dtype=dtype, device=device))
    rhs_init = torch.nn.Parameter(torch.randn(*batches, n_init_var_steps, n_dims, n_init_var_orders, dtype=dtype, device=device))
    steps = torch.nn.Parameter(torch.randn(*batches, n_steps - 1, dtype=dtype, device=device))
    u = ode_forward(coefficients, rhs_equation, rhs_init, steps)
    u.sum().backward()
    a = 0


if __name__ == '__main__':
    test()
