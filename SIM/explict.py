import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.data import DataLoader

class FNN(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList(
            [nn.Linear(architecture[i], architecture[i + 1]) for i in range(len(architecture) - 1)])
        self.activation = nn.ReLU()
    
    def forward(self, x):
        if x.ndim == 4:
            x = x.squeeze(1).flatten(start_dim=-2)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return F.softmax(self.layers[-1](x), dim=-1)
    
    def implicit_size(self):
        return sum(self.architecture[1:-1])
    
    def input_size(self):
        return self.architecture[0]
    
    def output_size(self):
        return self.architecture[-1]

def builder_explicit(cfg):
    if cfg.type == "fnn":
        model = FNN(cfg.architecture)
        optimizer = getattr(optim, cfg.optimizer)(model.parameters(), lr=cfg.lr)
        return model, optimizer

def build_SIM_matrix(cfg, ex_model, train_loader, device):
    import random
    from torch.utils.data import Subset, RandomSampler
    m = int(len(train_loader.dataset)*cfg.implicit.training_size)
    indices = random.sample(range(len(train_loader.dataset)), m)
    subset = Subset(train_loader.dataset, indices)
    sampler = RandomSampler(subset)
    state_loader = DataLoader(subset, sampler=sampler, batch_size=cfg.evaluation.bs, num_workers=2, drop_last=True)
    if cfg.explicit.type == "fnn":
        Z = torch.zeros(ex_model.implicit_size(), m)
        X = torch.zeros(ex_model.implicit_size(), m)
        Y = torch.zeros(ex_model.output_size(), m)
        U = torch.zeros(ex_model.input_size(), m)
        for i, (data, target) in enumerate(tqdm(state_loader)):
            data, target = data.to(device), target.to(device)
            length = data.shape[0]
            output = ex_model(data)
            temp = []
            temp_post = []
            data = data.squeeze(1).flatten(start_dim=-2)
            for layer in ex_model.layers[:-1]:
                data = layer(data)
                temp.append(data)
                temp_post.append(ex_model.activation(data))
            Z[:, i*length:(i+1)*length] = torch.concatenate(temp, dim=1).T
            X[:, i*length:(i+1)*length] = torch.concatenate(temp_post, dim=1).T
            Y[:, i*length:(i+1)*length] = output.T
    return [Z, X, Y, U], (ex_model.implicit_size(), ex_model.input_size(), ex_model.output_size()) 