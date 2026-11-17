import torch
from torch.autograd import Function
import sys
sys.path.append("lib/model/roi_pooling")
import roi_pooling_cuda  # 你构建出来的扩展模块

class RoIPoolFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, pooled_height, pooled_width, spatial_scale):
        ctx.feature_size = features.size()
        ctx.pooled_height = pooled_height
        ctx.pooled_width = pooled_width
        ctx.spatial_scale = spatial_scale
        ctx.save_for_backward(rois)

        output, argmax = roi_pooling_cuda.roi_pooling_forward(
            features, rois, spatial_scale, pooled_height, pooled_width
        )

        ctx.argmax = argmax
        return output

    @staticmethod
    def backward(ctx, grad_output):
        rois, = ctx.saved_tensors
        batch_size, channels, height, width = ctx.feature_size

        grad_input = roi_pooling_cuda.roi_pooling_backward(
            grad_output,
            rois,
            ctx.argmax,
            ctx.spatial_scale,
            batch_size,
            height,
            width,
            channels,
            ctx.pooled_height,
            ctx.pooled_width
        )

        # backward return values must match forward inputs count
        return grad_input, None, None, None, None
