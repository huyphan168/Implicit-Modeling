import torch
from .utils import transpose


class ImplicitPerturbation(torch.nn.Module):
    def forward(ctx, A, B, C, D, sigma_u, proxy=False):
        n, p = B.shape

        if not isinstance(sigma_u, torch.Tensor) or len(sigma_u.shape) == 0:
            sigma_u = sigma_u * torch.ones((p, 1), requires_grad=False)
            sigma_u = sigma_u.to(A.device)
        else:
            sigma_u = transpose(sigma_u)


        assert sigma_u.shape[0] == p, "shape inconsistency in sigma_u!"

        A = torch.abs(A)
        Bb = torch.abs(B @ sigma_u) if proxy else torch.abs(B) @ sigma_u
        C = torch.abs(C)
        Dd = torch.abs(D @ sigma_u) if proxy else torch.abs(D) @ sigma_u

        IAinv = torch.inverse(torch.eye(n).to(A.device) - A)

        #sigma_y = (C @ IAinv @ B + D) @ sigma_u
        sigma_y = C @ IAinv @ Bb + Dd
        #sigma_y = Dd

        return transpose(sigma_y)
