import torch
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
from .logger import Logger


def transpose(X):
    assert len(X.size()) == 2
    X = torch.transpose(X, -1, -2)
    return X


# def get_valid_accuracy(model, loss_fn, valid_dl, device):
#     from nn import ImplicitRobustLayer, ImplicitRobustLayerRank1FT
#     for xs, ys in valid_dl:
#         xs, ys = xs.to(device), ys.to(device)
#         if isinstance(model, (ImplicitRobustLayer, ImplicitRobustLayerRank1FT)):
#             pred, _ = model(xs, 0.0)
#         else:
#             pred = model(xs)
#         pred = pred if isinstance(pred, torch.Tensor) else torch.from_numpy(pred).to(device)
#         loss = loss_fn(pred, ys)
#         pred_i = np.argmax(pred.cpu().detach().numpy(), axis=-1)
#         correct = np.sum([1 if ys[i] == pred_i[i] else 0 for i in range(len(ys))])
#         return loss, correct/len(ys)


# def get_robust_accuracy(model, loss_fn, sigma, valid_dl, device):
#     from nn import ImplicitRobustLayer, ImplicitRobustLayerRank1FT
#     for xs, ys in valid_dl:
#         xs = torch.clamp(xs + torch.zeros_like(xs).normal_(std=sigma), min=0.0, max=1.0)
#         xs, ys = xs.to(device), ys.to(device)
#         if isinstance(model, (ImplicitRobustLayer, ImplicitRobustLayerRank1FT)):
#             pred, _ = model(xs, 0.0)
#         else:
#             pred = model(xs)
#         loss = loss_fn(pred, ys)
#         pred_i = np.argmax(pred.cpu().detach().numpy(), axis=-1)
#         correct = np.sum([1 if ys[i] == pred_i[i] else 0 for i in range(len(ys))])
#         return loss, correct / len(ys)


def get_ABCD_from_NN(model_state):
    w0 = model_state['0.weight'].cpu().numpy()
    w1 = model_state['2.weight'].cpu().numpy()

    n, p = w0.shape
    q, _ = w1.shape

    A = np.zeros((n, n))
    B = w0
    C = w1
    D = np.zeros((q, p))

    return A, B, C, D


def get_ABCD_from_NN_NBC(model_state):
    w0 = model_state['0.weight'].cpu().numpy()
    w1 = model_state['2.weight'].cpu().numpy()

    n, p = w0.shape
    q, _ = w1.shape

    A = np.zeros((n, n))
    B = w0
    C = w1
    D = np.zeros((q, p))

    return A, B, C, D


def get_ABCD_from_NN_784604010(model_state):
    dat = model_state

    B = sp.bmat([[sp.coo_matrix((40, 784))], [dat["0.weight"].cpu().numpy()]]).toarray()
    A = sp.bmat([[None, dat["2.weight"].cpu().numpy()], [sp.coo_matrix((60, 40)), None]]).toarray()
    C = sp.bmat([[dat["4.weight"].cpu().numpy(), sp.coo_matrix((10, 60))]]).toarray()
    D = np.zeros((10, 784))

    return A, B, C, D


def set_parameters_uniform(model, parameter=0.05, seed=None):
    if seed:
        print("using random seed: {}".format(seed))
        np.random.seed(seed)
    for name, param in model.named_parameters():
        print("setting weight {} of shape {} to be uniform(-{}, {})".format(name, param.shape, parameter, parameter))
        p = np.random.uniform(low=-parameter, high=parameter, size=param.shape)
        param.data.copy_(torch.from_numpy(p))
    return model


def RobustCrossEntropy(y_hat, sigma_y, y):
    if len(y.shape) >= 2:
        # TODO: deal with one-hot labels
        raise NotImplementedError()
    else:
        # y are indexes
        return F.cross_entropy(y_hat+sigma_y, y) + 2 * torch.mean(sigma_y[0, y])


def train(model, train_dl, valid_dl, optimizer, loss_fn, epochs, dirname, device=None):
    # load the model to GPU / CPU device
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger = Logger(printstr=["batch: {}. loss: {:.2f}, valid_loss/acc: {:.2f}/{}", "batch", "loss", "valid_loss", "valid_acc"],
                dir_name=dirname)

    for i in range(epochs):
        j = 0
        for xs, ys in train_dl:
            # forward step
            pred = model(xs.to(device))
            loss = loss_fn(pred, ys.to(device))

            # backward step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # log the performance and the model
            valid_res = get_valid_accuracy(model, loss_fn, valid_dl, device)
            log_dict = {
                "batch": j,
                "loss": loss,
                "valid_loss": valid_res[0],
                "valid_acc":valid_res[1]
            }
            logger.log(log_dict, model, "valid_acc")

            j+=1
        print("--------------epoch: {}. loss: {}".format(i, loss))
        pass
    return model, logger