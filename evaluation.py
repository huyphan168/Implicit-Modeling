import torch
import torch.nn as nn
import torch.nn.functional as F
from SIM.ImplicitModel import ImplicitModel
import numpy as np

def count_zeros(matrix):
    return np.count_nonzero(matrix == 0)/np.prod(matrix.shape)*100

def evaluate(cfg, weight_matrices, ex_model, test_loader, device):
    for method in cfg.baselines:
        if method == "explicit":
            ex_model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = ex_model(data)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)
            print('Explicit model: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
        elif method == "implicit":
            A,B,C,D = weight_matrices
            n = A.shape[0]
            q, p = D.shape[0], D.shape[1]
            implicit_model = ImplicitModel(n, cfg.bs, p, q)
            implicit_model.set_weights(A,B,C,D)
            implicit_model.to(device)
            implicit_model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    #TO:DO ugly work around
                    output = implicit_model(data.squeeze(1).flatten(start_dim=-2))
                    output = F.softmax(output, dim=-1)
                    test_loss += F.nll_loss(output, target, reduction='sum').item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
            test_loss /= len(test_loader.dataset)
            print('Implicit model: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))
            print("Sparsity of A: {0:.2f}%".format(count_zeros(A)))
            print("Sparsity of B: {0:.2f}%".format(count_zeros(B)))
            print("Sparsity of C: {0:.2f}%".format(count_zeros(C)))
            print("Sparsity of D: {0:.2f}%".format(count_zeros(D)))
            M1 = torch.cat([A,B], dim=1)
            M2 = torch.cat([C,D], dim=1)
            M = torch.cat([M1,M2], dim=0)
            print("Sparsity of M: {0:.2f}%".format(count_zeros(M)))
        else:
            raise NotImplementedError
