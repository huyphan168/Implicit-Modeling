import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor, Compose, Lambda
def load_data(cfg):
    name = cfg.evaluation.dataset
    if name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=Compose([ToTensor()]))
        testset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=Compose([ToTensor()]))
        trainloader = DataLoader(trainset, batch_size=cfg.evaluation.bs, shuffle=True, num_workers=2, drop_last=True)
        testloader = DataLoader(testset, batch_size=cfg.evaluation.bs, shuffle=False, num_workers=2)
        return trainloader, testloader