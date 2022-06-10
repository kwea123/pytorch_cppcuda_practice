import torch
import add_matrix_cuda


class AddMatrixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        output = add_matrix_cuda.forward(A, B)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        dA, dB = add_matrix_cuda.backward(grad_out)
        return dA, dB


if __name__=='__main__':
    f = AddMatrixFunction()
    A = torch.rand(100, 100, device='cuda:0').requires_grad_()
    B = torch.rand(100, 100, device='cuda:0')

    out = f.apply(A, B)
    print('model out:', out)
    loss = out.sum()
    loss.backward()
    print("A.grad:", A.grad)