import cvxpy as cp
import numpy as np
from omegaconf import DictConfig
import torch

class SparseOptimizer:
    def __init__(self, cfg, X, U, _params):
        self.cfg = cfg
        params = [param.squeeze(0) for param in _params]
        n = params[1].size()[0]
        p = params[2].size()[0]
        self.a = cp.Variable(n)
        self.b = cp.Variable(p)
        self.t = cp.Variable(n+p)
        self.s = cp.Variable(n+p)
        XU = torch.concat([X.T, U.T], dim=1).detach().numpy()
        obj = cp.Minimize(cfg.alpha * cp.sum(self.s) + cfg.lambda1*cp.norm(params[0].detach().numpy()-XU@cp.hstack([self.a, self.b])))
        # easy constraint!
        constraints = [self.t >= 0, self.s >= 0, 1>=self.t, cp.norm(self.a, 1) <= cfg.kappa]
        # rotated constraint - workaround
        constraints += [self.s[i] >= cp.quad_over_lin(self.a[i], self.t[i]) for i in range(n)]
        constraints += [self.s[i+n] >= cp.quad_over_lin(self.b[i], self.t[i+n]) for i in range(p)]
        self.problem = cp.Problem(obj, constraints)        
    
    def optimize(self):
        self.problem.solve(solver=cp.MOSEK)
        return self.a.value, self.b.value

class RobustOptimizer:
    def __init__(self, cfg, X, U, _params):
        self.cfg = cfg
        params = [param.squeeze(0) for param in _params]
        n = params[1].size()[0]
        p = params[2].size()[0]
        self.a = cp.Variable(n)
        self.b = cp.Variable(p)
        XU = torch.concat([X.T, U.T], dim=1).detach().numpy()
        obj = cp.Minimize(cp.norm(cp.hstack([self.a, self.b]), 1) + cfg.lambda1*cp.norm(params[0].detach().numpy()-XU@cp.hstack([self.a, self.b])))
        # easy constraint!
        constraints = [cp.norm(self.a, 1) <= cfg.kappa]
        self.problem = cp.Problem(obj, constraints)        
    
    def optimize(self):
        self.problem.solve(solver=cp.MOSEK)
        return self.a.value, self.b.value

def build_optimizer(cfg, X, U, params):
    if cfg.objective == "sparsity":
        return SparseOptimizer(cfg, X, U, params)
    if cfg.objective == "robustness":
        return RobustOptimizer(cfg, X, U, params)

if __name__ == "__main__":
    cfg = DictConfig({"implicit": 
                        {"workers": 32,
                         "kappa": 0.99,
                         "objective": "sparsity",
                         "lambda": [0.1, 0.1],
                         "training_size": 0.05}})
    SparseOptimizer(cfg, [])