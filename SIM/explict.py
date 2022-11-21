import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm 
from torch.utils.data import DataLoader
import numpy as np

class FNN(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList(
            [nn.Linear(architecture[i], architecture[i + 1], bias=False) for i in range(len(architecture) - 1)])
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
    
    def maxrownorm(self):
        #Get the largest max-row-sum of the weights among layers
        norms = []
        for layer in self.layers:
            # import ipdb; ipdb.set_trace()
            norms.append(np.linalg.norm(layer.weight.data.detach().cpu().numpy(), ord=np.inf))
        return max(norms)

    def rescale_weight(self):
        max_norm = self.maxrownorm()
        self.rescaled_layers = nn.ModuleList(
            [nn.Linear(self.architecture[i], self.architecture[i + 1], bias=False) for i in range(len(self.architecture) - 1)])
        for rescaled_layer, layer in zip(self.rescaled_layers, self.layers):
            rescaled_layer.weight.data = layer.weight.data / (2*max_norm)
            # rescaled_layer.weight.data = layer.weight.data
        print("max norm is", max_norm)

            # layer.bias.data = layer.bias.data / 1.5*self.maxrownorm()

def builder_explicit(cfg):
    if cfg.type == "fnn":
        model = FNN(cfg.architecture)
        optimizer = getattr(optim, cfg.optimizer)(model.parameters(), lr=cfg.lr)
        return model, optimizer

def build_SIM_matrix(cfg, ex_model, train_loader, device):
    import random
    from torch.utils.data import Subset
    m = int(len(train_loader.dataset)*cfg.implicit.training_size)
    indices = random.sample(range(len(train_loader.dataset)), m)
    subset = Subset(train_loader.dataset, indices)
    state_loader = DataLoader(subset, batch_size=cfg.evaluation.bs, num_workers=2)
    bs = cfg.evaluation.bs
    ex_model.rescale_weight()
    if cfg.explicit.type == "fnn":
        Z = torch.zeros(ex_model.implicit_size(), m)
        X = torch.zeros(ex_model.implicit_size(), m)
        Y = torch.zeros(ex_model.output_size(), m)
        inputs = []
        for i, (data, target) in enumerate(tqdm(state_loader)):
            data, target = data.to(device), target.to(device)
            length = data.shape[0]
            temp = []
            temp_post = []
            data = data.squeeze(1).flatten(start_dim=-2)
            inputs.append(data.T)
            for layer in ex_model.rescaled_layers[:-1]:
                data = layer(data)
                temp.append(data)
                data = ex_model.activation(data)
                temp_post.append(data)
            # output = F.softmax(ex_model.layers[-1](data), dim=-1)
            # output = F.softmax(ex_model.rescaled_layers[-1](data), dim=-1)
            output = ex_model.layers[-1](data)
            Z[:, i*bs: i*bs+length] = torch.cat(temp[::-1], dim=1).T
            X[:, i*bs: i*bs+length] = torch.cat(temp_post[::-1], dim=1).T
            Y[:, i*bs: i*bs+length] = output.T
        U = torch.cat(inputs, dim=-1)
        # import ipdb; ipdb.set_trace()
    return [Z, X, Y, U], (ex_model.implicit_size(), ex_model.input_size(), ex_model.output_size())