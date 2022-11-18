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
def SIM_element(cfg, X, U, params):
    # state_match, a, b = params
    optimizer = build_optimizer(cfg, X, U, params)
    return optimizer.optimize()
        
def SIM_training(cfg, states, shape):
    #Z = (nxm), X = (nxm), Y = (qxm)
    Z, X, Y, U = states
    Z.cpu()
    X.cpu()
    Y.cpu()
    U.cpu()
    n,p,q = shape
    A = torch.zeros(n, n)
    B = torch.zeros(n, p)
    C = torch.zeros(q, n)
    D = torch.zeros(q, p)
    #mini_params = (1xm)
    weight_AB = ProgressParallel(n_jobs=cfg.workers)(joblib.delayed(SIM_element)(cfg, X, U, mini_params) for mini_params in 
                        zip(torch.split(Z, 1, dim=0), torch.split(A, 1, dim=0), torch.split(B, 1, dim=0)))
    weight_CD = ProgressParallel(n_jobs=cfg.workers)(joblib.delayed(SIM_element)(cfg, X, U, mini_params) for mini_params in 
                        zip(torch.split(Y, 1, dim=0), torch.split(C, 1, dim=0), torch.split(D, 1, dim=0)))
    A = torch.concat([torch.tensor(ab[0]).unsqueeze(0) for ab in weight_AB])
    B = torch.concat([torch.tensor(ab[1]).unsqueeze(0) for ab in weight_AB])
    C = torch.concat([torch.tensor(cd[0]).unsqueeze(0) for cd in weight_CD])
    D = torch.concat([torch.tensor(cd[1]).unsqueeze(0) for cd in weight_CD])
    return A, B, C, D

def main() -> None:
    args = parse_arguments()
    cfg = OmegaConf.load(args.cfg)
    print("Building explict models ...")
    ex_model, ex_optimizer = builder_explicit(cfg.explicit)
    ex_criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ex_model.to(device)
    torch.save(ex_model.state_dict(), args.output_path + "explicit_model.pt")
    ex_criterion.to(device)
    print("Loading data ...")
    train_loader, test_loader = load_data(cfg)
    print("Training explicit ...")
    ex_model_trained = training_explicit(cfg.explicit, ex_model, ex_optimizer, ex_criterion, train_loader, device)
    print("Building state matrix ...")
    states, shape = build_SIM_matrix(cfg, ex_model_trained, train_loader, device)
    print("Training implicit ...")
    A,B,C,D = SIM_training(cfg.implicit, states, shape)
    #Evaluation both explicit and implicit models
    evaluate(cfg.evaluation, [A,B,C,D], ex_model_trained, test_loader, device)


if __name__ == "__main__":
    main()
