import torch
from torch.autograd import Function
import sys
sys.path.append("lib/model/roi_crop")
import roi_crop_cuda


class RoICropFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2):
        ctx.save_for_backward(input1, input2)
        output = roi_crop_cuda.forward(input1, input2)[0]
        # print(f"********** [DEBUG] input1 shape: {input1.shape}, input2 shape: {input2.shape}, output shape: {output.shape} **********")      # 观察用
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, input2 = ctx.saved_tensors
        grad_input1, grad_input2 = roi_crop_cuda.backward(input1, input2, grad_output)
        return grad_input1, grad_input2
