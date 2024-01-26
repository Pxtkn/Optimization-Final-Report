import time
from sklearn.datasets import load_svmlight_file
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

configs = {
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'need_record': True,
    'max_iter': 3000,
    'M': 0,
    'optimizer': 'SharpenedBFGS', # GD, BFGS, GreedyBFGS, SharpenedBFGS
    'dataset': 'colon-cancer',
    'dataset_configs': {
        'svmguide3': {
            'N': 1243,
            'd': 22,
            'mu': 1e-2,
            'data_path': './dataset/svmguide3.txt'
        },
        'w8a': {
            'N': 49749,
            'd': 300,
            'mu': 1e-4,
            'data_path': './dataset/w8a.txt'
        },
        'colon-cancer': {
            'N': 62,
            'd': 2000,
            'mu': 1e-5,
            'data_path': './dataset/colon-cancer'
        },
    },
}

class Objective(nn.Module):
    def __init__(self, data, lables):
        super(Objective, self).__init__()
        self.x = nn.Parameter(torch.ones(d) / (d * np.sqrt(d)))
        self.y = torch.from_numpy(lables).to(device)
        self.z = F.normalize(torch.from_numpy(data), dim=1).to(device)
        self.z = -self.y.reshape(-1, 1) * self.z

    def forward(self):
        t = torch.log(1 + torch.exp(self.z @ self.x))
        return torch.sum(t) / N + mu / 2 * torch.dot(self.x, self.x)

    def hessian(self):
        with torch.no_grad():
            t = torch.exp(self.z @ self.x).reshape(-1, 1)
            I = torch.eye(d)
            return self.z.T @ (self.z * t / (1 + t)**2) / N + mu * I

class GD(optim.Optimizer):
    def __init__(self, obj_func, need_record=False):
        self.obj_func = obj_func
        params = list(obj_func.parameters())
        super(GD, self).__init__(params, {})

        self.zero_grad()
        loss = self.obj_func()
        loss.backward()
        self.x = obj_func.x
        self.g = obj_func.x.grad.data.clone()

        self.k = 0
        self.need_record = need_record
        self.lambda_0 = self.newton_decrement()

    def step(self):
        # compute the gradient and update the parameters
        self.zero_grad()
        loss = self.obj_func()
        loss.backward()
        self.g = self.x.grad.data.clone()
        self.x.data = self.x.data - 1./L * self.g

        self.record(loss)
        return loss

    def record(self, loss):
        # update the iteration number and record
        self.k += 1
        if self.need_record:
            ratio = self.newton_decrement() / self.lambda_0
            writer.add_scalar('ratio', ratio, self.k)
            writer.add_scalar('loss', loss.item(), self.k)

    def newton_decrement(self):
        with torch.no_grad():
            g = self.g
            H = self.obj_func.hessian()
            return torch.sqrt(torch.dot(g, H @ g))
        
class BFGS(GD):
    def __init__(self, obj_func, need_record=False):
        super(BFGS, self).__init__(obj_func, need_record=need_record)
        self.B = torch.eye(d) * L

    def step(self):
        s = self.update_params()
        y, loss = self.update_grad()

        # update the Hessian approximation
        t = self.B @ s
        self.B = self.B - torch.outer(t, t) / torch.dot(t, s) + torch.outer(y, y) / torch.dot(y, s)
        
        self.record(loss)
        return loss

    def update_params(self):
        # update the parameters and return the difference
        s = -torch.inverse(self.B) @ self.g
        self.x.data = self.x.data + s
        return s
    
    def update_grad(self):
        # update the gradient and return the difference
        self.zero_grad()
        loss = self.obj_func()
        loss.backward()
        gk = self.x.grad.data.clone()
        y = gk - self.g
        self.g = gk
        return y, loss

class GreedyBFGS(BFGS):
    def __init__(self, obj_func, need_record=False):
        super(GreedyBFGS, self).__init__(obj_func, need_record=need_record)

    def step(self):
        s = self.update_params()
        y, loss = self.update_grad()

        # update the Hessian approximation
        B_bar = self.B
        H_new = self.obj_func.hessian()
        index = 0
        max_res = B_bar[0, 0] / H_new[0, 0]
        for i in range(1, d):
            res = B_bar[i, i] / H_new[i, i]
            if res > max_res:
                max_res = res
                index = i
        t = B_bar[:, index]
        y = H_new[:, index]
        self.B = B_bar - torch.outer(t, t) / B_bar[index, index] + torch.outer(y, y) / H_new[index, index]
        
        self.record(loss)
        return loss
    
class SharpenedBFGS(BFGS):
    def __init__(self, obj_func, need_record=False):
        super(SharpenedBFGS, self).__init__(obj_func, need_record=need_record)

    def step(self):
        H = self.obj_func.hessian()
        s = self.update_params()
        y, loss = self.update_grad()

        # update the Hessian approximation
        t = self.B @ s
        self.B = self.B - torch.outer(t, t) / torch.dot(t, s) + torch.outer(y, y) / torch.dot(y, s)

        r = torch.sqrt(torch.dot(s, H @ s))
        B_bar = (1 + 0.5 * M * r) * self.B
        H_new = self.obj_func.hessian()
        index = 0
        max_res = B_bar[0, 0] / H_new[0, 0]
        for i in range(1, d):
            res = B_bar[i, i] / H_new[i, i]
            if res > max_res:
                max_res = res
                index = i
        t = B_bar[:, index]
        y = H_new[:, index]
        self.B = B_bar - torch.outer(t, t) / B_bar[index, index] + torch.outer(y, y) / H_new[index, index]
        
        self.record(loss)
        return loss

if __name__ == '__main__':
    device = configs['device']
    need_record = configs['need_record']
    max_iter = configs['max_iter']
    M = configs['M']

    dataset = configs['dataset']
    dataset_configs = configs['dataset_configs'][dataset]
    N = dataset_configs['N']
    d = dataset_configs['d']
    mu = dataset_configs['mu']
    data_path = dataset_configs['data_path']
    optim_name = configs['optimizer']

    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)

    data, labels = load_svmlight_file(data_path)
    data = data.todense()

    L = mu + 0.25
    obj_func = Objective(data, labels)

    record_path = './records/' + dataset + '/' + optim_name
    if need_record:
        writer = SummaryWriter(record_path)

    if optim_name == 'GD':
        optimizer = GD(obj_func, need_record)
    elif optim_name == 'BFGS':
        optimizer = BFGS(obj_func, need_record)
    elif optim_name == 'GreedyBFGS':
        optimizer = GreedyBFGS(obj_func, need_record)
    elif optim_name == 'SharpenedBFGS':
        optimizer = SharpenedBFGS(obj_func, need_record)
    else:
        raise NotImplementedError

    for iteration in range(max_iter):
        loss = optimizer.step()

        if iteration % 10 == 0:
            lambda_k = optimizer.newton_decrement()
            print("Iteration {}: loss = {:.6f}, lambda = {:.3e}, ratio = {:.3e}".format(
                iteration, loss, lambda_k, lambda_k / optimizer.lambda_0))
