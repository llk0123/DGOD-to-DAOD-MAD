# import torch
# import nms_cuda  # 确保已使用 setup.py / CMake 编译为 .so 并在 PYTHONPATH 中
#
# def test_normal_case():
#     torch.cuda.set_device(4)
#     print("==> Testing: normal NMS case")
#
#     N = 1000
#     boxes = torch.zeros((N, 5), dtype=torch.float32, device='cuda')
#     boxes[:, 0] = torch.rand(N) * 100  # x1
#     boxes[:, 1] = torch.rand(N) * 100  # y1
#     boxes[:, 2] = boxes[:, 0] + torch.rand(N, device=boxes.device) * 20  # x2
#     boxes[:, 3] = boxes[:, 1] + torch.rand(N, device=boxes.device) * 20  # y2
#     boxes[:, 4] = torch.rand(N, device=boxes.device)  # score
#
#     keep, num = nms_cuda.nms_forward(boxes, iou_threshold=0.5)
#     print(f"→ Keep shape: {keep.shape}, num: {num.item()}")
#
# def test_zero_boxes():
#     torch.cuda.set_device(4)
#     print("==> Testing: zero boxes")
#
#     boxes = torch.empty((0, 5), dtype=torch.float32, device='cuda')
#     try:
#         keep, num = nms_cuda.nms_forward(boxes, 0.5)
#         print(f"→ Passed. keep: {keep}, num: {num}")
#     except Exception as e:
#         print(f"→ Failed with exception: {e}")
#
# def test_cpu_input():
#     torch.cuda.set_device(4)
#     print("==> Testing: CPU input (should fail)")
#
#     boxes = torch.zeros((10, 5), dtype=torch.float32)  # CPU tensor
#     try:
#         keep, num = nms_cuda.nms_forward(boxes, 0.5)
#         print("→ ERROR: Should have raised an exception")
#     except RuntimeError as e:
#         print(f"→ Correctly failed: {e}")
#
# def test_bad_shape():
#     torch.cuda.set_device(4)
#     print("==> Testing: bad shape input")
#
#     boxes = torch.zeros((10, 4), dtype=torch.float32, device='cuda')  # missing score
#     try:
#         keep, num = nms_cuda.nms_forward(boxes, 0.5)
#         print("→ ERROR: Should have failed due to shape")
#     except RuntimeError as e:
#         print(f"→ Correctly failed: {e}")
#
# def test_nms_nan_in_input():
#     torch.cuda.set_device(4)
#     print("==> Testing: input with NaN")
#
#     boxes = torch.zeros((10, 5), dtype=torch.float32, device='cuda')
#     boxes[:, 0:4] = torch.rand(10, 4) * 50
#     boxes[:, 4] = torch.rand(10)
#     boxes[3, 0] = float('nan')
#
#     try:
#         keep, num = nms_cuda.nms_forward(boxes, 0.5)
#         print(f"→ keep shape: {keep.shape}, num: {num.item()}")
#     except Exception as e:
#         print(f"→ Failed gracefully: {e}")
#
# if __name__ == "__main__":
#     torch.cuda.init()
#     test_normal_case()
#     test_zero_boxes()
#     test_cpu_input()
#     test_bad_shape()
#     test_nms_nan_in_input()


import torch
import nms_cuda

def test_normal_case():
    print("==> Testing: normal NMS case")
    torch.cuda.set_device(3)

    N = 1000
    boxes = torch.zeros((N, 5), dtype=torch.float32, device='cuda:4')

    # 👇 所有临时张量也都放到同一个 device 上
    boxes[:, 0] = torch.rand(N, device=boxes.device) * 100  # x1
    boxes[:, 1] = torch.rand(N, device=boxes.device) * 100  # y1
    boxes[:, 2] = boxes[:, 0] + torch.rand(N, device=boxes.device) * 20  # x2
    boxes[:, 3] = boxes[:, 1] + torch.rand(N, device=boxes.device) * 20  # y2
    boxes[:, 4] = torch.rand(N, device=boxes.device)  # score

    iou_threshold = 0.5
    keep, num_out = nms_cuda.nms_forward(boxes, iou_threshold)

    print("==> Output:")
    print("keep indices:", keep[:10].cpu().numpy())
    print("num_out:", num_out.item())

if __name__ == "__main__":
    test_normal_case()


"""极简测试代码，仅使用 3 个 boxes，绝不会爆显存，核心用于验证 kernel 执行路径是否安全"""
# import torch
# import nms_cuda  # 请确保你正确 import 到修复后的模块
#
# def test_tiny_nms():
#     torch.cuda.set_device(3)
#
#     boxes = torch.tensor([
#         [10.0, 10.0, 20.0, 20.0, 0.9],  # box1
#         [12.0, 12.0, 22.0, 22.0, 0.8],  # box2
#         [50.0, 50.0, 60.0, 60.0, 0.7],  # box3
#     ], dtype=torch.float32).cuda()
#
#     print("Input boxes:")
#     print(boxes)
#
#     keep, num = nms_cuda.nms_forward(boxes, 0.5)
#
#     print("Keep indices:", keep.cpu().tolist())
#     print("Num kept:", num.item())
#
# test_tiny_nms()
