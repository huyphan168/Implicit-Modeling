import torch
from .ImplicitFunction import ImplicitFunctionTriu
from .ImplicitModel import ImplicitModel
from .ImplicitPerturbation import ImplicitPerturbation
from torch import nn
from .utils import transpose


class ImplicitRobustModel(ImplicitModel):
    def __init__(self, n, m, p, q, f=ImplicitFunctionTriu, X0=None, no_D=False):
        super(ImplicitRobustModel, self).__init__(n, m, p, q, f, X0, no_D=no_D)
        self.perturb = ImplicitPerturbation()


    def forward(self, U, sigma_u, proxy=False):
        Y = super(ImplicitRobustModel, self).forward(U)
        sigma = self.perturb(self.A, self.B, self.C, self.D, sigma_u, proxy=proxy)
        return Y, sigma#torch.zeros_like(sigma)#sigma


class ImplicitRobustModelRank1FT(ImplicitRobustModel):
    # Finetune model with rank 1 parameters
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n = self.A.shape[0]

        self.u = nn.Parameter(torch.randn(n)/n)
        self.v = nn.Parameter(torch.randn(n)/n)

    def forward(self, U, sigma_u, proxy=False):

        A = self.A.detach() + torch.ger(self.rescale(self.u),  self.rescale(self.v))
        B = self.B.detach()
        C = self.C.detach()
        D = self.D.detach()

        U = transpose(U)
        X = self.f.apply(A, B, self.X0, U)
        Y = transpose(C @ X + D @ U)

        sigma = self.perturb(A, B, C, D, sigma_u, proxy=proxy)
        return Y, sigma#torch.zeros_like(sigma)#sigma

    @staticmethod
    def rescale(u):
        size = torch.sqrt(torch.sum(u ** 2))
        if size > 1:
            u.data.copy_(u / size)
        return u


class ImplicitRobustModelTriuFT(ImplicitRobustModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n = self.A.shape[0]

        self.A_p = nn.Parameter(torch.randn(n, n)/n)

    def forward(self, U, sigma_u, proxy=False):

        self.A_p.data.copy_(self.A_p.triu(1))

        A = self.A.detach() + self.A_p
        B = self.B.detach()
        C = self.C.detach()
        D = self.D.detach()

        U = transpose(U)
        X = self.f.apply(A, B, self.X0, U)
        Y = transpose(C @ X + D @ U)

        sigma = self.perturb(A, B, C, D, sigma_u, proxy=proxy)
        return Y, sigma#torch.zeros_like(sigma)#sigma
