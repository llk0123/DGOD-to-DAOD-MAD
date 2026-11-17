# model/utils/dmp.py
import torch
import math
import torch.nn as nn

class DynamicMaskingProcess(nn.Module):
    """
    动态掩蔽过程（Dynamic Masking Process, DMP）
    用于在训练阶段对输入图像进行随机块遮挡，增强模型的鲁棒性和泛化能力。
    """

    def __init__(self, patch_size=8, max_mask_rate=0.5):
        """
        Args:
            patch_size (int): 每个掩蔽块的边长，默认 8。
            max_mask_rate (float): 最大掩蔽比例，默认 0.5。
        """
        super(DynamicMaskingProcess, self).__init__()
        self.patch_size = patch_size
        self.max_mask_rate = max_mask_rate

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        执行动态掩蔽操作。
        Args:
            images (Tensor): 输入图像张量 (B, C, H, W)
        Returns:
            Tensor: 掩蔽后的图像
        """
        # 测试阶段不做处理
        if not self.training:
            return images

        B, C, H, W = images.shape
        P = self.patch_size
        device = images.device

        assert H % P == 0 and W % P == 0, f"H={H}, W={W} 必须能被 patch_size={P} 整除"

        num_patches_h = H // P
        num_patches_w = W // P
        total_patches = num_patches_h * num_patches_w

        # 克隆副本以便原图不被修改
        masked_images = images.clone()

        for i in range(B):
            # 随机掩蔽率 [0, max_mask_rate]
            mask_rate = torch.empty(1, device=device).uniform_(0.0, self.max_mask_rate).item()
            num_mask = math.ceil(mask_rate * total_patches)

            # 随机选择掩蔽块索引
            perm = torch.randperm(total_patches, device=device)
            mask_indices = perm[:num_mask]

            patch_rows = mask_indices // num_patches_w
            patch_cols = mask_indices % num_patches_w

            for r, c in zip(patch_rows.tolist(), patch_cols.tolist()):
                h_start = r * P
                w_start = c * P
                masked_images[i, :, h_start:h_start + P, w_start:w_start + P] = 0.0

        return masked_images
