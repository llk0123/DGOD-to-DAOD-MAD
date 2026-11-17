import torch
from torch.autograd import Function
import roi_crop_cuda  # .so 模块

class ROICropFunction(Function):
    @staticmethod
    def forward(ctx, input, grid):
        output = roi_crop_cuda.forward(input, grid)[0]
        # output = roi_crop_cuda.forward(input, grid)
        ctx.save_for_backward(input, grid)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid = ctx.saved_tensors
        grad_input, grad_grid = roi_crop_cuda.backward(input, grid, grad_output)
        return grad_input, grad_grid
