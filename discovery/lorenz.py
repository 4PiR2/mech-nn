import os

import numpy as np
from scipy.integrate import odeint
from scipy.special import logit
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import discovery.basis as B
import discovery.plot as P
import extras.logger as logger
from extras.source import write_source_files, create_log_dir

from solver.ode_layer import ODEINDLayer as ODEINDLayer0
from solver.ode_layer import ODEINDLayer


class LorenzDataset(Dataset):
    def __init__(self, STEP, n_step_per_batch=100, n_step=1000):
        self.n_step_per_batch = n_step_per_batch
        self.n_step = n_step
        self.end = n_step * STEP
        x_train = self.generate()  # ndarray (n_step, 3)
        self.x_train = torch.tensor(x_train, dtype=torch.float64)

        # Create basis for some stats. Actual basis is in the model
        basis, basis_vars = B.create_library(x_train, polynomial_order=2, use_trig=False, constant=True)
        # ndarray (n_step, 10), list (10)
        self.basis = torch.tensor(basis)
        self.basis_vars = basis_vars
        self.n_basis = self.basis.shape[1]  # 10

    def generate(self):
        rho = 28.
        sigma = 10.
        beta = 8. / 3.

        def f(state, t):
            x, y, z = state
            return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

        state0 = [1., 1., 1.]
        time_steps = np.linspace(0., self.end, self.n_step)
        x_train = odeint(f, state0, time_steps)
        return x_train

    def __len__(self):
        return self.n_step - self.n_step_per_batch

    def __getitem__(self, idx):
        i = idx
        d = self.x_train[i:i+self.n_step_per_batch]
        return i, d


class Model(nn.Module):
    def __init__(self, bs, n_basis, n_step_per_batch, device=None, **kwargs):
        super().__init__()

        self.order = 2  # state dimension
        self.bs = bs  # batch size
        self.n_ind_dim = 3
        self.n_step_per_batch = n_step_per_batch  # 50
        self.n_basis = n_basis  # 10

        self.init_xi = torch.randn(1, self.n_basis, self.n_ind_dim, dtype=torch.float64, device=device)  # (1, 10, 3)

        self.mask = torch.ones_like(self.init_xi, device=device)

        # Step size is fixed. Make this a parameter for learned step
        self.step_size = logit(1e-2) * torch.ones(1, 1, 1)  # [[[-4.5951]]]
        self.xi = nn.Parameter(self.init_xi)
        self.param_in = nn.Parameter(torch.randn(1, 64))

        self.ode0 = ODEINDLayer0(
            bs=bs, order=self.order, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch, solver_dbl=True,
            double_ret=True, n_iv=1, n_iv_steps=1,  gamma=.05, alpha=0, **kwargs,
        )
        self.ode = ODEINDLayer(
            bs=bs, order=self.order, n_ind_dim=self.n_ind_dim, n_step=self.n_step_per_batch, solver_dbl=True,
            double_ret=True, n_iv=1, n_iv_steps=1,  gamma=.05, alpha=0, **kwargs,
        )

        self.param_net = nn.Sequential(
            nn.Linear(64, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_basis * self.n_ind_dim)
        )

        self.net = nn.Sequential(
            nn.Linear(self.n_step_per_batch * self.n_ind_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, self.n_step_per_batch * self.n_ind_dim)
        )
    
    def reset_params(self):
        # reset basis weights to random values
        self.xi.data = torch.randn_like(self.init_xi)

    def update_mask(self, mask):
        self.mask *= mask
    
    def get_xi(self):
        xi = self.param_net(self.param_in).reshape(self.init_xi.shape)  # (1, 10, 3)
        return xi

    def forward(self, net_iv):
        # apply mask
        xi = self.get_xi()
        xi = self.mask * xi
        _xi = xi  # (1, 10, 3)
        xi = xi.repeat(self.bs, 1, 1)  # (bs, 10, 3)

        var = self.net(net_iv.reshape(self.bs, -1)).reshape(self.bs, self.n_step_per_batch, self.n_ind_dim)
        # (bs, 50, 3)

        # create basis
        var_basis, _ = B.create_library_tensor_batched(var, polynomial_order=2, use_trig=False, constant=True)
        # (bs, 50, 10)

        rhs = var_basis @ xi  # (bs, 50, 3)
        rhs = rhs.permute(0, 2, 1)    # (bs, 3, 50)

        z = torch.zeros(1, self.n_ind_dim, 1, 1).type_as(net_iv)
        o = torch.ones(1, self.n_ind_dim, 1, 1).type_as(net_iv)

        coeffs = torch.cat([z, o, z], dim=-1)  # (1, 3, 1, 3)
        coeffs = coeffs.repeat(self.bs, 1, self.n_step_per_batch, 1)  # (bs, 3, 50, 3)

        steps = self.step_size.type_as(net_iv).sigmoid().repeat(self.bs, self.n_ind_dim, self.n_step_per_batch-1)

        x0, x1, x2, eps, steps = self.ode(coeffs, rhs, var[:, 0], steps)

        x0 = x0.permute(0, 2, 1)

        return x0, steps, eps, var, _xi


def print_eq(model, basis_vars, L, stdout=False):
    # print learned equation
    xi = model.get_xi()
    repr_dict = B.basis_repr(xi * model.mask, basis_vars)
    code = []
    for k, v in repr_dict.items():
        L.info(f'{k} = {v}')
        if stdout:
            print(f'{k} = {v}')
        code.append(f'{v}')
    return code


def simulate(gen_code, T, STEP):
    # simulate learned equation
    def f(state, t):
        x0, x1, x2 = state

        dx0 = eval(gen_code[0])
        dx1 = eval(gen_code[1])
        dx2 = eval(gen_code[2])

        return dx0, dx1, dx2
        
    state0 = [1., 1., 1.]
    time_steps = np.linspace(0., T * STEP, T)

    x_sim = odeint(f, state0, time_steps)
    return x_sim


def main():
    log_dir, run_id = create_log_dir(root='logs')
    # write_source_files(log_dir)
    L = logger.setup(log_dir, stdout=False)

    STEP = 0.01
    T = 10000
    n_step_per_batch = 50
    batch_size = 512
    # weights less than threshold (absolute) are set to 0 after each optimization step.
    threshold = 0.1

    ds = LorenzDataset(STEP, n_step=T, n_step_per_batch=n_step_per_batch)
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # plot train data
    P.plot_lorenz(ds.x_train, os.path.join(log_dir, 'train.pdf'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Model(bs=batch_size, n_basis=ds.n_basis, n_step_per_batch=n_step_per_batch, device=device)
    model.double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    """Optimize and threshold cycle"""
    model.reset_params()

    max_iter = 10
    for step in range(max_iter):
        print(f'Optimizer iteration {step}/{max_iter}')

        # threshold
        if step > 0:
            xi = model.get_xi()
            mask = (xi.abs() > threshold).float()

            L.info(xi)
            L.info(xi * model.mask)
            L.info(model.mask)
            L.info(model.mask * mask)

        # simulate and plot
        code = print_eq(model, ds.basis_vars, L, stdout=True)
        x_sim = simulate(code, T, STEP)  # (10000, 3)
        P.plot_lorenz(x_sim, os.path.join(log_dir, f'sim_{step}.pdf'))

        # set mask
        if step > 0:
            model.update_mask(mask)
            model.reset_params()

        nepoch = 400
        with tqdm(total=nepoch) as pbar:
            for epoch in range(nepoch):
                pbar.update(1)
                for i, (_, batch_in) in enumerate(train_loader):
                    batch_in = batch_in.to(device)

                    x0, _, eps, var, xi = model(batch_in)

                    x_loss = (x0 - batch_in).pow(2).mean()
                    loss = x_loss + (var - batch_in).pow(2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                xi = xi.detach().cpu().numpy()
                meps = eps.max().item()
                L.info(f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} ')
                print(f'basis\n {xi}')
                pbar.set_description(f'run {run_id} epoch {epoch}, loss {loss.item()} max eps {meps} xloss {x_loss} ')
    print_eq(model, ds.basis_vars, L, stdout=False)


if __name__ == '__main__':
    main()
