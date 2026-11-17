# import torch
# from torch.autograd import Function
# import sys
# sys.path.append("lib/model/roi_align")
# import roi_align_cuda  # 你编译出来的 .so 会被当作模块引入
#
#
# class RoIAlignFunction(Function):
#     @staticmethod
#     def forward(ctx, features, rois, aligned_height, aligned_width, spatial_scale):
#         ctx.save_for_backward(rois)
#         ctx.spatial_scale = spatial_scale
#         ctx.feature_size = features.size()
#         ctx.aligned_height = aligned_height
#         ctx.aligned_width = aligned_width
#
#         output = roi_align_cuda.forward(features, rois, aligned_height, aligned_width, spatial_scale)
#         return output
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         rois, = ctx.saved_tensors
#         aligned_height = ctx.aligned_height
#         aligned_width = ctx.aligned_width
#         spatial_scale = ctx.spatial_scale
#         feature_size = ctx.feature_size
#
#         grad_input = roi_align_cuda.backward(
#             grad_output.contiguous(), rois,
#             aligned_height, aligned_width, spatial_scale
#         )
#         return grad_input, None, None, None, None


import torch
from torch.autograd import Function
import sys
sys.path.append("lib/model/roi_align")
import roi_align_cuda  # 你编译出来的 .so 会被当作模块引入

class RoIAlignFunction(Function):
    @staticmethod
    def forward(ctx, features, rois, aligned_height, aligned_width, spatial_scale):
        ctx.save_for_backward(features, rois)  # 保存features用于backward
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = features.size()
        ctx.aligned_height = aligned_height
        ctx.aligned_width = aligned_width

        # 修改参数顺序：spatial_scale 应该在 aligned_height 之前
        output = roi_align_cuda.forward(
            features,
            rois,
            spatial_scale,  # 注意：spatial_scale 在前
            aligned_height,  # aligned_height 在后
            aligned_width
        )[0]  # 注意：可能需要 [0] 来获取列表中的第一个元素

        return output

    @staticmethod
    def backward(ctx, grad_output):
        features, rois = ctx.saved_tensors  # 获取保存的tensors
        aligned_height = ctx.aligned_height
        aligned_width = ctx.aligned_width
        spatial_scale = ctx.spatial_scale
        feature_size = ctx.feature_size

        # # 同样调整参数顺序
        # grad_input = roi_align_cuda.backward(
        #     grad_output.contiguous(),
        #     rois,
        #     spatial_scale,  # spatial_scale 在前
        #     aligned_height,
        #     aligned_width,
        #     feature_size[0],  # batch_size
        #     feature_size[1],  # num_channels
        #     feature_size[2],  # data_height
        #     feature_size[3]  # data_width
        # )[0]  # 可能需要 [0]


        # 根据 C++ 函数签名，需要传入这些参数：
        # top_grad, rois, spatial_scale, batch_size, channels, height, width, aligned_height, aligned_width
        grad_input = roi_align_cuda.backward(
            grad_output.contiguous(),  # top_grad
            rois,                      # rois
            spatial_scale,             # spatial_scale
            feature_size[0],           # batch_size
            feature_size[1],           # channels
            feature_size[2],           # height
            feature_size[3],           # width
            aligned_height,            # aligned_height
            aligned_width              # aligned_width
        )



        # 返回梯度：features的梯度，rois的梯度(None)，以及其他参数的梯度(都是None)
        return grad_input, None, None, None, None
