import cvxpy as cp
import numpy as np
from omegaconf import DictConfig
import torch

class SparseOptimizer:
    def __init__(self, cfg, X, U, _params, verbose, wellposed):
        self.cfg = cfg
        params = [param for param in _params]
        n = params[1].shape[1]
        p = params[2].shape[1]
        self.a = cp.Variable(n)
        self.b = cp.Variable(p)
        self.t = cp.Variable(n+p)
        self.s = cp.Variable(n+p)
        XU = np.concatenate([X.T, U.T], axis=1)
        # import ipdb; ipdb.set_trace()
        obj = cp.Minimize(cfg.alpha * cp.sum(self.s) + cfg.lambda1*cp.norm(params[0][0]-XU@cp.hstack([self.a, self.b]))**2)
        # easy constraint!
        constraints = [self.t >= 0, self.s >= 0, 1>=self.t]
        if wellposed:
             constraints.append(cp.norm(self.a, 1) <= cfg.kappa)
        # rotated constraint - workaround
        constraints += [self.s[i] >= cp.quad_over_lin(self.a[i], self.t[i]) for i in range(n)]
        constraints += [self.s[i+n] >= cp.quad_over_lin(self.b[i], self.t[i+n]) for i in range(p)]
        self.problem = cp.Problem(obj, constraints)   
        self.verbose = verbose
    
    def optimize(self):
        self.problem.solve(solver=cp.MOSEK, verbose=self.verbose)
        # if self.verbose:
        #     print(self.b.value)
        return self.a.value, self.b.value

class RobustOptimizer:
    def __init__(self, cfg, X, U, _params, verbose, wellposed):
        self.cfg = cfg
        params = [param for param in _params]
        n = params[1].shape[1]
        p = params[2].shape[1]
        self.a = cp.Variable(n)
        self.b = cp.Variable(p)
        XU = np.concatenate([X.T, U.T], axis=1)
        # import ipdb; ipdb.set_trace()
        obj = cp.Minimize(cp.norm(cp.hstack([self.a, self.b]), 1) + cfg.lambda1*cp.norm(params[0][0]-XU@cp.hstack([self.a, self.b])))
        # obj = cp.Minimize(cfg.lambda1*cp.norm(params[0][0]-XU@cp.hstack([self.a, self.b])))
        # easy constraint!
        if wellposed:
            constraints = [cp.norm(self.a, 1) <= cfg.kappa]
            # constraints = []
        else:
            constraints = []
        self.problem = cp.Problem(obj, constraints)     
        self.verbose=verbose   
    
    def optimize(self):
        self.problem.solve(solver=cp.MOSEK, verbose=self.verbose)
        return self.a.value, self.b.value

def build_optimizer(cfg, X, U, params, verbose, wellposed):
    verbose = True if verbose == 0 else False
    if cfg.objective == "sparsity":
        return SparseOptimizer(cfg, X, U, params, verbose, wellposed)
    if cfg.objective == "robustness":
        return RobustOptimizer(cfg, X, U, params, verbose, wellposed)

if __name__ == "__main__":
    cfg = DictConfig({"implicit": 
                        {"workers": 32,
                         "kappa": 0.99,
                         "objective": "sparsity",
                         "lambda": [0.1, 0.1],
                         "training_size": 0.05}})
    SparseOptimizer(cfg, [])