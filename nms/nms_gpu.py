import torch
import sys
sys.path.append("lib/model/nms")
import nms_cuda

def nms_gpu(dets, thresh):
    """
    Args:
        dets: Tensor of shape (N, 5) - [x1, y1, x2, y2, score]
        thresh: float - IoU threshold

    Returns:
        keep: Tensor of indices of kept boxes
    """
    # return nms_cuda.nms_forward(dets, thresh)
    keep, _ = nms_cuda.nms_forward(dets, thresh)  # unpack两个返回值
    return keep