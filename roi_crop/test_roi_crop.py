# import torch
# import roi_crop_cuda  # 你编译出来的 .so 被 Python import 为这个模块
# from roi_crop_function import ROICropFunction
#
# def test_forward_backward():
#     B, H, W, C = 2, 64, 64, 128
#     N = 4
#     out_H, out_W = 32, 32
#
#     input = torch.randn(B, H, W, C, device='cuda', dtype=torch.float32, requires_grad=True)
#     grid = torch.randn(N, out_H, out_W, 2, device='cuda', dtype=torch.float32, requires_grad=True)
#
#     # Forward
#     output = ROICropFunction.apply(input, grid)
#     print("Forward OK:", output.shape)
#
#     # Backward
#     loss = output.sum()
#     loss.backward()
#
#     print("Backward OK:")
#     print("  input.grad:", input.grad.shape, "nan:", torch.isnan(input.grad).any().item())
#     print("  grid.grad :", grid.grad.shape, "nan:", torch.isnan(grid.grad).any().item())
#
# if __name__ == "__main__":
#     torch.cuda.set_device(0)
#     test_forward_backward()


# deepseek给的测试代码
import torch
from roi_crop_function import ROICropFunction


def test_roi_crop():
    # 测试参数配置
    batch_size = 2
    height, width = 64, 64
    channels = 128
    num_rois = 4
    crop_size = (32, 32)

    # 生成测试数据 (保持BHWC格式)
    # 修改后（NCHW）
    feature_map = torch.randn(
        batch_size,
        channels,  # C调整到第二维度
        height,
        width,
        device='cuda',
        requires_grad=True
    )

    # 生成归一化网格坐标 (范围[-1,1])
    rois_grid = torch.rand(
        num_rois,
        crop_size[0],
        crop_size[1],
        2,
        device='cuda'
    ) * 2 - 1  # 归一化到[-1,1]
    rois_grid.requires_grad_(True)

    # 前向传播测试
    cropped_feat = ROICropFunction.apply(feature_map, rois_grid)
    print("[前向传播] 输出形状:", cropped_feat.shape)  # 应得 (4,128,32,32)

    # 反向传播测试
    loss = cropped_feat.sum()
    loss.backward()

    # 梯度有效性验证
    print("\n[反向传播] 梯度检查:")
    print(f"特征图梯度形状: {feature_map.grad.shape} | 存在NaN: {torch.isnan(feature_map.grad).any().item()}")
    print(f"网格梯度形状: {rois_grid.grad.shape} | 存在NaN: {torch.isnan(rois_grid.grad).any().item()}")


if __name__ == "__main__":
    torch.cuda.set_device(0)
    test_roi_crop()

