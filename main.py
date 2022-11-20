from omegaconf import OmegaConf
import argparse
import torch
import torch.nn as nn
from SIM.explict import builder_explicit, build_SIM_matrix
from SIM.data_utils import load_data
from SIM.optimization import build_optimizer
from evaluation import evaluate
from tqdm import tqdm 
import joblib
import os.path as osp
import numpy as np
import pickle as pkl
class ProgressParallel(joblib.Parallel):
    def __call__(self, *args, **kwargs):
        with tqdm() as self._pbar:
            return joblib.Parallel.__call__(self, *args, **kwargs)
    
    def print_progress(self):
        self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()

def parse_arguments():
    parser = argparse.ArgumentParser("State-Driven Implicit Deep Learning")
    parser.add_argument("--cfg", type=str, default="configs/sparsity")
    parser.add_argument("--output-path", type=str, default="results/")
    args = parser.parse_args()
    return args

def training_explicit(cfg, ex_model, ex_optimizer, ex_criterion, train_loader, device):
    for epoch in range(cfg.epoch):
        ex_model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            ex_optimizer.zero_grad()
            output = ex_model(data)
            loss = ex_criterion(output, target)
            loss.backward()
            ex_optimizer.step()
            #get accuracy of predictions
            prediction_label = torch.argmax(output, dim=1)
            acc = (prediction_label==target).sum().item()/len(data)
            if (batch_idx*len(data)) % cfg.log_interval == 0:
                print('Train Epoch: {} [{}/{}]\tLoss: {}\tAccuracy: {} %'.format(
                    epoch+1, batch_idx * len(data), len(train_loader.dataset), round(loss.item(), 2), round(100*acc,2)))
    return ex_model
def SIM_element(cfg, X, U, params, verbose, wellposed):
    # state_match, a, b = params
    optimizer = build_optimizer(cfg, X, U, params, verbose, wellposed)
    return optimizer.optimize()
        
def SIM_training(cfg, states, shape):
    #Z = (nxm), X = (nxm), Y = (qxm)
    Z, X, Y, U = states
    Z = Z.cpu().detach().numpy()
    X = X.cpu().detach().numpy()
    Y= Y.cpu().detach().numpy()
    U = U.cpu().detach().numpy()
    n,p,q = shape
    A = torch.zeros(n, n)
    B = torch.zeros(n, p)
    C = torch.zeros(q, n)
    D = torch.zeros(q, p)
    #mini_params = (1xm)
    print(U.sum())
    # import ipdb; ipdb.set_trace()
    weight_AB = ProgressParallel(n_jobs=cfg.workers)(joblib.delayed(SIM_element)(cfg, X, U, mini_params, idx, True) for idx, mini_params in enumerate(
                        zip(np.split(Z, Z.shape[0], axis=0), torch.split(A, 1, dim=0), torch.split(B, 1, dim=0))))
    weight_CD = ProgressParallel(n_jobs=cfg.workers)(joblib.delayed(SIM_element)(cfg, X, U, mini_params, idx, False) for idx, mini_params in enumerate(
                        zip(np.split(Y, Y.shape[0], axis=0), torch.split(C, 1, dim=0), torch.split(D, 1, dim=0))))
    A = torch.cat([torch.tensor(ab[0]).unsqueeze(0) for ab in weight_AB])
    B = torch.cat([torch.tensor(ab[1]).unsqueeze(0) for ab in weight_AB])
    C = torch.cat([torch.tensor(cd[0]).unsqueeze(0) for cd in weight_CD])
    D = torch.cat([torch.tensor(cd[1]).unsqueeze(0) for cd in weight_CD])
    return A, B, C, D

def main() -> None:
    args = parse_arguments()
    cfg = OmegaConf.load(args.cfg)
    print("Building explict models ...")
    ex_model, ex_optimizer = builder_explicit(cfg.explicit)
    ex_criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex_criterion.to(device)
    print("Loading data ...")
    train_loader, test_loader = load_data(cfg)
    print("Training explicit ...")
    if osp.exists(args.output_path + "explicit_model.pt"):
        ex_model_weight = torch.load(args.output_path + "explicit_model.pt")
        ex_model.load_state_dict(ex_model_weight)
        ex_model.to(device)
    else:
        ex_model.to(device)
        ex_model = training_explicit(cfg.explicit, ex_model, ex_optimizer, ex_criterion, train_loader, device) 
        torch.save(ex_model.state_dict(), args.output_path + "explicit_model.pt")
    print("Building state matrix ...")
    states, shape = build_SIM_matrix(cfg, ex_model, train_loader, device)
    
    if cfg.implicit.evaluate:
        print("Loading optimized weight matrices")
        with open("results/states.pkl", "rb") as f:
            unpickler = pkl.Unpickler(f)
            states_load = unpickler.load()
            states_load = states_load["states"]
            A, B, C, D = states_load
            
            print(np.sum((np.abs(D.numpy())<1e-4).astype(int))/np.prod(D.shape))
            print(np.mean(C.cpu().detach().numpy()))
            print(np.max(C.cpu().detach().numpy()))
    else:
        print("Training implicit ...")
        A,B,C,D = SIM_training(cfg.implicit, states, shape)
        import pickle
        with open(osp.join(args.output_path, "states.pkl"), "wb") as f:
            pickle.dump({"states": [A,B,C,D]}, f)
    
    #Prunning
    B[np.abs(B)<1e-3] = 0
    A[np.abs(A)<1e-4] = 0
    # C = torch.zeros_like(C)
    D[np.abs(D)<2e-8] = 0
    #Evaluation both explicit and implicit models
            
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,15), dpi=300)
    M1 = torch.cat([A, B],dim=-1).numpy()
    M2 = torch.cat([C, D],dim=-1).numpy()
    M = np.concatenate([M1, M2], axis=0)
    plt.spy(M, markersize=0.01, color='black')
    plt.savefig("states.png")
    evaluate(cfg.evaluation, [A,B,C,D], ex_model, test_loader, device)
    # evaluate(cfg.evaluation, [A,B,C,D], ex_model, train_loader, device)

if __name__ == "__main__":
    main()
