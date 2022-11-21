import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import ToTensor, Compose, Lambda
import pickle
from pathlib import Path
import torchvision
import torchvision.transforms as transforms
import os
import gzip
import requests
import torch
from torch.utils.data import TensorDataset, DataLoader

def load_data(cfg):
    name = cfg.evaluation.dataset
    if name == "FashionMNIST":
        trainset = torchvision.datasets.FashionMNIST(
            root='./data', train=True, download=True, transform=Compose([ToTensor()]))
        testset = torchvision.datasets.FashionMNIST(
            root='./data', train=False, download=True, transform=Compose([ToTensor()]))
        trainloader = DataLoader(trainset, batch_size=cfg.evaluation.bs, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=cfg.evaluation.bs, shuffle=False, num_workers=2)
        return trainloader, testloader
    if name == "MNIST":
        trainset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=Compose([ToTensor()]))
        testset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=Compose([ToTensor()]))
        trainloader = DataLoader(trainset, batch_size=cfg.evaluation.bs, shuffle=True, num_workers=2)
        testloader = DataLoader(testset, batch_size=cfg.evaluation.bs, shuffle=False, num_workers=2)
        return trainloader, testloader
        # return mnist_load(cfg.evaluation.bs)

# def mnist_download():
#     PATH = "data/mnist"
#     os.makedirs(PATH, exist_ok=True)
#     URL = "http://deeplearning.net/data/mnist/"
#     FILENAME = "mnist.pkl.gz"
#     if not os.path.exists(os.path.join(PATH, FILENAME)):
#         content = requests.get(URL + FILENAME).content
#         with open(os.path.join(PATH, FILENAME), "wb") as f:
#             f.write(content)
        
# def mnist_load(train_bs, valid_bs=10000):
#     mnist_download()
#     PATH = "data/mnist"
#     FILENAME = "mnist.pkl.gz"
#     with gzip.open(os.path.join(PATH, FILENAME), "rb") as f:
#         ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")
#         x_train, y_train, x_valid, y_valid, x_test, y_test = map(
#             torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)
#         )
#     train_ds = TensorDataset(x_train, y_train)
#     train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
#     valid_ds = TensorDataset(x_valid, y_valid)
#     valid_dl = DataLoader(valid_ds, batch_size=valid_bs, shuffle=True)
#     return train_dl, valid_dl
# __file_path = os.path.abspath(__file__)
# __proj_dir = "/".join(str.split(__file_path, "/")[:-2]) + "/"
# DATA_PATH = Path(__proj_dir)

# def mnist_download():
#     PATH = DATA_PATH / "data" / "mnist"
#     PATH.mkdir(parents=True, exist_ok=True)
#     URL = "http://deeplearning.net/data/mnist/"
#     FILENAME = "mnist.pkl.gz"
#     if not (PATH / FILENAME).exists():
#         content = requests.get(URL + FILENAME).content
#         (PATH / FILENAME).open("wb").write(content)
# def mnist_load(train_bs, valid_bs=10000):
#     mnist_download()
#     PATH = DATA_PATH / "data" / "mnist"
#     FILENAME = "mnist.pkl.gz"
#     with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
#         ((x_train, y_train), (x_valid, y_valid), (x_test, y_test)) = pickle.load(f, encoding="latin-1")
#         x_train, y_train, x_valid, y_valid, x_test, y_test = map(
#             torch.tensor, (x_train, y_train, x_valid, y_valid, x_test, y_test)
#         )
#     train_ds = TensorDataset(x_train, y_train)
#     train_dl = DataLoader(train_ds, batch_size=train_bs, shuffle=True)
#     valid_ds = TensorDataset(x_valid, y_valid)
#     valid_dl = DataLoader(valid_ds, batch_size=valid_bs, shuffle=True)
#     return train_dl, valid_dl