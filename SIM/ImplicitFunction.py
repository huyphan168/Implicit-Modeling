import torch
import numpy as np
from torch.autograd import Function
from .utils import transpose

# AA: function used for trying to parallelize for loop
# def computation(a,idx):
#     X, Y, A, U, grad_output = a[0], a[1], a[2], a[3], a[4]
#     n, m = X.size()
#     x = X[:, idx:idx + 1]
#     y = Y[:, idx:idx + 1]

#     dp = ImplicitFunction.dphi(y).view(n)

#     D = torch.diag(dp)
#     Jinv = torch.inverse(torch.eye(n) - D @ A)
#     r = torch.transpose(torch.transpose(grad_output[:,idx:idx+1],-1,-2) @ Jinv,-1,-2).view(n)
#     z = r * dp

#     grad_A = torch.ger(z.view(n), x.view(n))
#     grad_B = torch.ger(z.view(n), U[:,idx])

#     #grad_A = 1
#     #grad_B = 1

#     return (grad_A, grad_B)


class ImplicitFunction(Function):
    @staticmethod
    def forward(ctx, A, B, X0, U):
        with torch.no_grad():
            X, err, status = ImplicitFunction.inn_pred(A, B @ U)
        ctx.save_for_backward(A, B, X, U)
        if status not in "converged":
            print("Picard iterations not converging!", err, status)
        return X

    @staticmethod
    def backward(ctx, *grad_outputs):
        A, B, X, U = ctx.saved_tensors

        grad_output = grad_outputs[0]
        assert grad_output.size() == X.size()

        # n, m = X.size()
        # p, _ = U.size()

        DPhi = ImplicitFunction.dphi(A @ X + B @ U)
        V, err, status = ImplicitFunction.inn_pred_grad(A.T, DPhi * grad_output, DPhi)
        if status not in "converged":
            print("Gradient iteration not converging!", err, status)
        grad_A = V @ X.T
        grad_B = V @ U.T
        grad_U = B.T @ V

        '''Old Analytic Solution
        # involving pytorch
        #grad_output_np = grad_output.cpu().numpy()
        A_np = A.clone().detach().cpu().numpy()
        B_np = B.clone().detach().cpu().numpy()
        X_np = X.clone().detach().cpu().numpy()
        U_np = U.clone().detach().cpu().numpy()
        Y = ImplicitFunction.phi(A @ X + B @ U)
        Dphi_np = ImplicitFunction.dphi(Y).cpu().numpy()
        ###################

        grad_A_np = np.zeros(A_np.shape)
        grad_B_np = np.zeros(B_np.shape)
        grad_U_np = np.zeros(U_np.shape)

        for i in range(m):
            x = X_np[:, i:i + 1]
            u = U_np[:, i:i + 1]
            gradout = grad_output_np[:, i:i + 1]

            # now not involving pytorch
            dp = Dphi_np[:, i:i + 1]
            ###################
            D = np.diag(np.squeeze(dp))
            Jinv = np.linalg.inv(np.eye(n) - D @ A_np)
            r = np.transpose(np.transpose(gradout) @ Jinv)
            z = r * dp

            grad_A_np += np.outer(z, x)
            grad_B_np += np.outer(z, u)
            grad_U_np[:, i:i + 1] = np.transpose(B_np) @ z

        #########################################
        # I now convert your computation to torch tensor and feed to backward function.
        if grad_A_np is not None:
            grad_A = torch.FloatTensor(grad_A_np).to(U.device)
            grad_B = torch.FloatTensor(grad_B_np).to(U.device)
            grad_U = torch.FloatTensor(grad_U_np).to(U.device)
        '''

        return (grad_A, grad_B, torch.zeros_like(X), grad_U)

    @staticmethod
    def phi(X):
        return torch.clamp(X, min=0)


    # TODO: Make universal
    @staticmethod
    def dphi(X):
        grad = X.clone().detach()
        grad[X <= 0] = 0
        grad[X > 0] = 1

        return grad

    @staticmethod
    def inn_pred(A, Z, mitr=300, tol=3e-6):
        X = torch.zeros_like(Z)
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            X_new = ImplicitFunction.phi(A @ X + Z)
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status

    @staticmethod
    def inn_pred_grad(AT, Z, DPhi, mitr=300, tol=3e-6):
        X = torch.zeros_like(Z)
        err = 0
        status = 'max itrs reached'
        for i in range(mitr):
            X_new = DPhi * (AT @ X) + Z
            err = torch.norm(X_new - X, np.inf)
            if err < tol:
                status = 'converged'
                break
            X = X_new
        return X, err, status

    @staticmethod
    def get_jacobian(y, x, v=None):
        if callable(y):
            x.requires_grad_(True)
            with torch.enable_grad():
                y = y(x)

        if v:   # output vTJ
            return torch.autograd.grad(y, x, v, retain_graph=True, only_inputs=True)[0]
        else:
            jacobian = torch.zeros(y.size()[0], x.size()[0], dtype=x.dtype).to(x.device)
            n = y.size()[0]
            for i in range(n):
                jacobian[i] = transpose(
                    torch.autograd.grad(y, x, torch.eye(n).to(x.device)[:,i:i+1], retain_graph=True, only_inputs=True)[0])
            return jacobian

    @staticmethod
    def get_invjacobian_prod(y, x, vT):
        # TODO: computation acceleration

        J = ImplicitFunction.get_jacobian(y, x)
        J_inv = torch.inverse(J)

        return vT @ J_inv


# class Sum:
#     def __init__(self,n,p):
#         self.A = np.zeros((n,n))
#         self.B = np.zeros((n,p))
#         self.lock = thread.allocate_lock()
#         self.count = 0

#     def add(self,valueA,valueB):
#         self.count += 1
#         self.lock.acquire()
#         self.A += valueA
#         self.B += valueB
#         self.lock.release()


class ImplicitFunctionTriu(ImplicitFunction):
    def forward(ctx, A, B, X0, U):
        A = A.triu_(1)
        return ImplicitFunction.forward(ctx, A, B, X0, U)

    def backward(ctx, *grad_outputs):
        grad_A, grad_B, x, u = ImplicitFunction.backward(ctx, *grad_outputs)
        return (grad_A.triu(1), grad_B, x, u)


class ImplicitFunctionInf(ImplicitFunction):
    def forward(ctx, A, B, X0, U):

        # project A on |A|_inf=v
        v = 0.95
        #v = 0.2
        # TODO: verify and speed up
        A_np = A.clone().detach().cpu().numpy()
        x = np.abs(A_np).sum(axis=-1)
        for idx in np.where(x > v)[0]:
            # read the vector
            a_orig = A_np[idx, :]
            a_sign = np.sign(a_orig)
            a_abs = np.abs(a_orig)
            a = np.sort(a_abs)

            s = np.sum(a) - v
            l = float(len(a))
            for i in range(len(a)):
                # proposal: alpha <= a[i]
                if s / l > a[i]:
                    s -= a[i]
                    l -= 1
                else:
                    break
            alpha = s / l
            a = a_sign * np.maximum(a_abs - alpha, 0)
            # verify
            assert np.isclose(np.abs(a).sum(), v)
            # write back
            A_np[idx, :] = a
        # 0.0
        #A_np = np.zeros_like(A_np)
        A.data.copy_(torch.tensor(A_np, dtype=A.dtype, device=A.device))

        return ImplicitFunction.forward(ctx, A, B, X0, U)

    def backward(ctx, *grad_outputs):
        grad_A, grad_B, x, u = ImplicitFunction.backward(ctx, *grad_outputs)
        return (grad_A, grad_B, x, u)
