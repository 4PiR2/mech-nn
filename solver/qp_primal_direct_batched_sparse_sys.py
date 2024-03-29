import torch
from torch.autograd import Function

from solver.lp_sparse_forward_diff import ODESYSLP


def solve_kkt2(A, L, d, _b, gamma):
    """
        Solve min x'Gx + d'x
            Ax = b

            g := d
            h := -b
            p := x*
            G := gamma*I
    """
    
    At = A.transpose(1, 2)

    _b = _b.unsqueeze(2)
    d = d.unsqueeze(2)

    rhs1 = A @ d - gamma * _b
    lam = torch.cholesky_solve(rhs1, L)

    z = (At @ lam - d) / gamma
    
    return z, lam


def QPFunction(ode: ODESYSLP, n_step=100, order=2, n_iv=2, gamma=1, alpha=1, DEVICE='cuda', double_ret=True):

    class QPFunctionFn(Function):
        @staticmethod
        #def forward(ctx, coeffs, rhs, iv_rhs, derivative_A):
        def forward(ctx, eq_A, rhs, iv_rhs, derivative_A):
        #def forward(ctx, coeffs, rhs, iv_rhs):
            #bs = coeffs.shape[0]
            bs = rhs.shape[0]
            #ode.build_ode(coeffs, rhs, iv_rhs, derivative_A)
            ode.build_ode(eq_A, rhs, iv_rhs, derivative_A)
            #ode.build_ode(coeffs, rhs, iv_rhs, None)
            
            At = ode.AG.to_dense()

            ub = ode.ub

            b = torch.zeros(bs, ode.num_vars).type_as(rhs)
            #print("c ", c.dtype, c.shape, At.shape)
            #minimize gamma*eps^2 +alpha*eps
            b[:, 0] = alpha
            
            c = ub.type_as(rhs)
            A = At.transpose(1, 2)
            AAt = A @ At

            print((A != 0.).sum(dim=[-2, -1]) / (A.size(-2) * A.size(-1)))
            print((AAt != 0.).sum(dim=[-2, -1]) / (AAt.size(-2) * AAt.size(-1)))

            L, info = torch.linalg.cholesky_ex(AAt, upper=False)

            x, y = solve_kkt2(A, L, c, -b, gamma)

            x = x.squeeze(2)
            y = y.squeeze(2)

            ctx.save_for_backward(A, L, x, y)
            
            if not double_ret:
                y = y.float()
            return y

        @staticmethod
        def backward(ctx, dl_dzhat):
            A, L, _x, _y = ctx.saved_tensors
            
            bs = dl_dzhat.shape[0]
            m = ode.num_constraints

            z = torch.zeros(bs, m).type_as(_x)

            _dx,_dnu = solve_kkt2(A, L, z, -dl_dzhat, gamma)
            
            _dx, _dnu = -_dx,-_dnu

            db = _dx[:, :ode.num_added_equation_constraints]
            db = -db.squeeze(-1)

            if ode.n_iv == 0:
                div_rhs = None
            else:
                div_rhs = -_dx[:, ode.num_added_equation_constraints:ode.num_added_equation_constraints + ode.num_added_initial_constraints].squeeze(2)

            # step gradient
            dD = ode.sparse_grad_derivative_constraint(_dx,_y)
            dD = dD + ode.sparse_grad_derivative_constraint(_x,_dnu)

            # eq grad
            dA = ode.sparse_grad_eq_constraint(_dx,_y)
            dA = dA + ode.sparse_grad_eq_constraint(_x,_dnu)

            if not double_ret:
                dA = dA.float()
                db =db.float()
                div_rhs = div_rhs.float() if div_rhs is not None else None
                dD = dD.float()
            
            return dA, db, div_rhs, dD

    return QPFunctionFn.apply
