import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.autograd import Variable
from torchvision.ops import nms
import numpy as np
import sys
sys.path.append('../..')
from model.utils.config import cfg
from model.rpn.rpn_new import _RPN
from model.roi_pooling.modules.roi_pool import _RoIPooling
from model.roi_crop.modules.roi_crop import _RoICrop
from model.roi_align.modules.roi_align import RoIAlignAvg
from model.rpn.proposal_target_layer_cascade import _ProposalTargetLayer
import time
import pdb
from model.utils.net_utils import _smooth_l1_loss, _crop_pool_layer, _affine_grid_gen, _affine_theta
# DA
from model.faster_rcnn_MV_3.DA import _ImageDA, _InstanceDA, _ImageDA_noGRL, _InstanceDA_En, _ImageDA_res, _ImageDA_ST, _InstanceDA_ST
from model.faster_rcnn_MV_3.AC import AutoDecoder, AutoEncoder, InsEncoder, InsDecoder, ImgEncoder_1, ImgEncoder_2, ImgEncoder_3, ImgDecoder_1, ImgDecoder_2, ImgDecoder_3


def pseudo_label(rois, cls_prob, bbox_pred, im_info, score_thresh=0.5, nms_thresh=0.3, class_agnostic=False):
    """
    Generate pseudo labels from RoIs, classification scores, and bbox_pred
    Returns: pseudo_boxes [N, 5], pseudo_labels [N]
    """

    # 处理batch维度
    if rois.dim() == 3:
        batch_size = rois.size(0)
        # 将rois展平为2D: [batch_size * num_rois, 5]
        rois = rois.view(-1, 5)

    boxes = rois[:, 1:5]  # remove batch idx, shape: [N, 4]
    batch_indices = rois[:, 0]  # 保存batch索引
    scores = cls_prob.detach()  # shape: [N, num_classes]
    bbox_deltas = bbox_pred.detach()  # shape: [N, 4] or [N, num_classes*4]

    # 添加详细的调试信息
    # print(f"\n[DEBUG Pseudo Label Generation]")
    # print(f"Total RoIs: {scores.size(0)}")
    # print(
    #     f"Score statistics - Max: {scores.max().item():.4f}, Mean: {scores.mean().item():.4f}, Min: {scores.min().item():.4f}")

    # 分析每个类别的最高分（排除背景类）
    # if scores.size(1) > 1:  # 确保有多个类别
    #     max_scores_per_class, _ = scores[:, 1:].max(dim=0)  # 排除背景类
        # print(f"Max score per class (excluding background): {max_scores_per_class.cpu().numpy()}")

    # 统计不同阈值下的检测数
    # print(f"Number of detections per score threshold:")
    # for thresh in [0.1, 0.3, 0.5, 0.7, 0.9]:
        # print(f"  > {thresh}: {(scores > thresh).sum().item()}")

    # Apply bbox regression
    if not class_agnostic:
        num_rois = scores.size(0)
        num_classes = scores.size(1)

        # 使用repeat_interleave
        boxes = boxes.repeat_interleave(num_classes, dim=0)  # [N*num_classes, 4]
        batch_indices = batch_indices.repeat_interleave(num_classes, dim=0)  # [N*num_classes]

        # 重塑bbox_deltas为[N*num_classes, 4]
        bbox_deltas = bbox_deltas.view(num_rois, num_classes, 4)
        bbox_deltas = bbox_deltas.contiguous().view(-1, 4)

        # 计算预测框
        pred_boxes = boxes + bbox_deltas

        # 展平scores并创建标签
        scores = scores.view(-1)  # [N*num_classes]
        labels = torch.arange(num_classes).repeat(num_rois).to(rois.device)
    else:
        # class_agnostic情况
        pred_boxes = boxes + bbox_deltas
        scores, labels = torch.max(scores, dim=1)

    # Clip boxes (需要按batch处理)
    if im_info.dim() > 1:
        # 对每个batch分别处理
        for b in range(int(batch_indices.max().item()) + 1):
            mask = batch_indices == b
            if mask.any():
                im_h, im_w = int(im_info[b][0]), int(im_info[b][1])
                pred_boxes[mask, 0::2] = pred_boxes[mask, 0::2].clamp(0, im_w - 1)
                pred_boxes[mask, 1::2] = pred_boxes[mask, 1::2].clamp(0, im_h - 1)
    else:
        im_h, im_w = int(im_info[0]), int(im_info[1])
        pred_boxes[:, 0::2] = pred_boxes[:, 0::2].clamp(0, im_w - 1)
        pred_boxes[:, 1::2] = pred_boxes[:, 1::2].clamp(0, im_h - 1)

    # Apply score threshold
    keep = torch.where(scores > score_thresh)[0]

    # print(f"\n[DEBUG] Before NMS: {keep.numel()} boxes above threshold {score_thresh}")

    if keep.numel() == 0:
        # print("[DEBUG] No boxes above threshold, returning empty results")
        return torch.zeros((0, 5), device=rois.device), torch.zeros((0,), dtype=torch.long, device=rois.device)

    boxes_keep = pred_boxes[keep]
    scores_keep = scores[keep]
    labels_keep = labels[keep]
    batch_indices_keep = batch_indices[keep]

    # 按batch进行NMS
    final_boxes = []
    final_labels = []
    final_batch_indices = []

    for b in range(int(batch_indices_keep.max().item()) + 1):
        mask = batch_indices_keep == b
        if mask.any():
            b_boxes = boxes_keep[mask]
            b_scores = scores_keep[mask]
            b_labels = labels_keep[mask]

            # print(f"  Batch {b}: {mask.sum().item()} boxes before NMS")

            # 只对非背景类进行NMS
            non_bg_mask = b_labels > 0
            if non_bg_mask.any():
                b_boxes_fg = b_boxes[non_bg_mask]
                b_scores_fg = b_scores[non_bg_mask]
                b_labels_fg = b_labels[non_bg_mask]

                # NMS
                keep_idx = nms(b_boxes_fg, b_scores_fg, nms_thresh)

                # print(f"  Batch {b}: {keep_idx.numel()} foreground boxes after NMS")

                if keep_idx.numel() > 0:
                    final_boxes.append(b_boxes_fg[keep_idx])
                    final_labels.append(b_labels_fg[keep_idx])
                    final_batch_indices.extend([b] * keep_idx.numel())

                    # 打印前3个框的信息
                    # num_to_show = min(3, keep_idx.numel())
                    # for i in range(num_to_show):
                    #     idx = keep_idx[i]
                    #     print(f"    Box {i}: cls={b_labels_fg[idx].item()}, score={b_scores_fg[idx].item():.3f}, "
                    #           f"coords=[{b_boxes_fg[idx][0]:.1f}, {b_boxes_fg[idx][1]:.1f}, "
                    #           f"{b_boxes_fg[idx][2]:.1f}, {b_boxes_fg[idx][3]:.1f}]")

    if len(final_boxes) == 0:
        # print("[DEBUG] No boxes after NMS, returning empty results")
        return torch.zeros((0, 5), device=rois.device), torch.zeros((0,), dtype=torch.long, device=rois.device)

    # 合并所有batch的结果
    boxes_nms = torch.cat(final_boxes, dim=0)
    labels_nms = torch.cat(final_labels, dim=0)
    batch_indices_nms = torch.tensor(final_batch_indices, device=rois.device).unsqueeze(1).float()

    # 构造pseudo_boxes，包含batch索引
    pseudo_boxes = torch.cat([batch_indices_nms, boxes_nms], dim=1)

    # print(f"[DEBUG Pseudo Label] Final: {pseudo_boxes.size(0)} pseudo labels generated")

    return pseudo_boxes, labels_nms


class _fasterRCNN(nn.Module):
    """ faster RCNN """

    def __init__(self, classes, class_agnostic):
        super(_fasterRCNN, self).__init__()
        self.classes = classes
        self.n_classes = len(classes)
        self.class_agnostic = class_agnostic
        # loss
        self.RCNN_loss_cls = 0
        self.RCNN_loss_bbox = 0

        # define rpn
        self.RCNN_rpn = _RPN(self.dout_base_model)
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)
        self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0 / 16.0)

        self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        self.RCNN_roi_crop = _RoICrop()

        # DA_s12
        self.RCNN_imageDA = _ImageDA(self.dout_base_model)
        self.RCNN_instanceDA = _InstanceDA()

        # DA_MV
        self.RCNN_imageDA_en1 = _ImageDA(128)  # 域判别头
        self.RCNN_imageDA_en2 = _ImageDA(128)  # 域判别头
        self.RCNN_imageDA_en3 = _ImageDA(128)  # 域判别头
        # self.RCNN_imageDA_en = _ImageDA_res(128)    # 域判别头
        # self.RCNN_imageDA_en1 = self.RCNN_imageDA_en    # 域判别头
        # self.RCNN_imageDA_en2 = self.RCNN_imageDA_en    # 域判别头
        # self.RCNN_imageDA_en3 = self.RCNN_imageDA_en    # 域判别头

        # self.RCNN_imageDA_en1 = _ImageDA_res(128)    # 域判别头
        # self.RCNN_imageDA_en2 = _ImageDA_res(128)    # 域判别头
        # self.RCNN_imageDA_en3 = _ImageDA_res(128)    # 域判别头

        self.RCNN_imageDA_en1_ST = _ImageDA(128)
        self.RCNN_imageDA_en2_ST = _ImageDA(128)
        self.RCNN_imageDA_en3_ST = _ImageDA(128)

        # self.RCNN_insDA_or = _InstanceDA()
        self.RCNN_insDA_en1 = _InstanceDA_En()
        self.RCNN_insDA_en2 = _InstanceDA_En()
        self.RCNN_insDA_en3 = _InstanceDA_En()
        # self.RCNN_insDA_en = _InstanceDA_En()
        # self.RCNN_insDA_en1 = self.RCNN_insDA_en
        # self.RCNN_insDA_en2 = self.RCNN_insDA_en
        # self.RCNN_insDA_en3 = self.RCNN_insDA_en
        self.RCNN_insDA_en1_ST = _InstanceDA_En()
        self.RCNN_insDA_en2_ST = _InstanceDA_En()
        self.RCNN_insDA_en3_ST = _InstanceDA_En()

        # DA_ST
        self.RCNN_imageDA_ST = _ImageDA_ST(self.dout_base_model)
        self.RCNN_instanceDA_ST = _InstanceDA_ST()
        self.da_img_loss_fn = nn.CrossEntropyLoss()
        self.da_ins_loss_fn = nn.BCELoss()
        # self.ce_loss = nn.CrossEntropyLoss()

        self.RCNN_insDA_loss = nn.BCELoss()

        self.consistency_loss = torch.nn.MSELoss(size_average=False)

        self.ln_img = nn.LayerNorm(normalized_shape=[10, 19])
        self.ln_ins = nn.LayerNorm(normalized_shape=[512])

        # AutoEncoder
        self.AutoEn_1 = ImgEncoder_1()
        self.AutoDe_1 = ImgDecoder_1()
        self.InsEn_1 = InsEncoder()
        self.InsDe_1 = InsDecoder()

        self.AutoEn_2 = ImgEncoder_2()
        self.AutoDe_2 = ImgDecoder_2()
        self.InsEn_2 = InsEncoder()
        self.InsDe_2 = InsDecoder()

        self.AutoEn_3 = ImgEncoder_3()
        self.AutoDe_3 = ImgDecoder_3()
        self.InsEn_3 = InsEncoder()
        self.InsDe_3 = InsDecoder()

        self.mse_loss = nn.MSELoss()
        self.upsample = torch.nn.Upsample(size=(40, 76), mode='bilinear')

    def forward(self,   im_data_s1, im_info_s1, gt_boxes_s1, num_boxes_s1,
                        im_data_s2, im_info_s2, gt_boxes_s2, num_boxes_s2,
                        im_data_t, im_info_t):

        batch_size = im_data_s1.size(0)

        im_info_s1 = im_info_s1.data  # (size1,size2, image ratio(new image / source image) )
        gt_boxes_s1 = gt_boxes_s1.data
        num_boxes_s1 = num_boxes_s1.data
        domain_label_s1 = Variable(torch.FloatTensor([1.] * batch_size).cuda())

        im_info_s2 = im_info_s2.data  # (size1,size2, image ratio(new image / source image) )
        gt_boxes_s2 = gt_boxes_s2.data
        num_boxes_s2 = num_boxes_s2.data
        domain_label_s2 = Variable(torch.FloatTensor([0.] * batch_size).cuda())

        # feed image data to base model to obtain base feature map
        # 将来自两个不同源（s1/s2）的图像，分别提取基础特征，并进入 RPN，产生候选区域（RoI proposals）
        base_feat_s1 = self.RCNN_base(im_data_s1)
        base_feat_s2 = self.RCNN_base(im_data_s2)
        # base_feat_t = self.RCNN_base(im_data_t)

        # feed base feature map tp RPN to obtain rois
        self.RCNN_rpn.train()
        rois_s1, rpn_loss_cls_s1, rpn_loss_bbox_s1 = self.RCNN_rpn(base_feat_s1, im_info_s1, gt_boxes_s1, num_boxes_s1)
        rois_s2, rpn_loss_cls_s2, rpn_loss_bbox_s2 = self.RCNN_rpn(base_feat_s2, im_info_s2, gt_boxes_s2, num_boxes_s2)

        # if it is training phrase, then use ground trubut bboxes for refining
        if self.training:
            # s1
            roi_data_s1 = self.RCNN_proposal_target(rois_s1, gt_boxes_s1, num_boxes_s1)
            rois_s1, rois_label_s1, rois_target_s1, rois_inside_ws_s1, rois_outside_ws_s1 = roi_data_s1

            rois_label_s1 = Variable(rois_label_s1.view(-1).long())
            rois_target_s1 = Variable(rois_target_s1.view(-1, rois_target_s1.size(2)))
            rois_inside_ws_s1 = Variable(rois_inside_ws_s1.view(-1, rois_inside_ws_s1.size(2)))
            rois_outside_ws_s1 = Variable(rois_outside_ws_s1.view(-1, rois_outside_ws_s1.size(2)))

            # s2
            roi_data_s2 = self.RCNN_proposal_target(rois_s2, gt_boxes_s2, num_boxes_s2)
            rois_s2, rois_label_s2, rois_target_s2, rois_inside_ws_s2, rois_outside_ws_s2 = roi_data_s2

            rois_label_s2 = Variable(rois_label_s2.view(-1).long())
            rois_target_s2 = Variable(rois_target_s2.view(-1, rois_target_s2.size(2)))
            rois_inside_ws_s2 = Variable(rois_inside_ws_s2.view(-1, rois_inside_ws_s2.size(2)))
            rois_outside_ws_s2 = Variable(rois_outside_ws_s2.view(-1, rois_outside_ws_s2.size(2)))

        else:
            rois_label_s1 = None
            rois_target_s1 = None
            rois_inside_ws_s1 = None
            rois_outside_ws_s1 = None
            rpn_loss_cls_s1 = 0
            rpn_loss_bbox_s1 = 0

            rois_label_s2 = None
            rois_target_s2 = None
            rois_inside_ws_s2 = None
            rois_outside_ws_s2 = None
            rpn_loss_cls_s2 = 0
            rpn_loss_bbox_s2 = 0

        rois_s1 = Variable(rois_s1)
        rois_s2 = Variable(rois_s2)
        # do roi pooling based on predicted rois

        # roi pooling
        # 提取出的 RoI 特征分别送入检测头，进行目标分类和边界框回归。
        if cfg.POOLING_MODE == 'align':
            pooled_feat_s1 = self.RCNN_roi_align(base_feat_s1, rois_s1.view(-1, 5))
            pooled_feat_s2 = self.RCNN_roi_align(base_feat_s2, rois_s2.view(-1, 5))
        elif cfg.POOLING_MODE == 'pool':
            pooled_feat_s1 = self.RCNN_roi_pool(base_feat_s1, rois_s1.view(-1, 5))
            pooled_feat_s2 = self.RCNN_roi_pool(base_feat_s2, rois_s2.view(-1, 5))

        # print("[DEBUG] pooled_feat_s1.shape:", pooled_feat_s1.shape)      # 观察用
        # print("[DEBUG] pooled_feat_s2.shape:", pooled_feat_s2.shape)
        # feed pooled features to top model
        pooled_feat_s1 = self._head_to_tail(pooled_feat_s1)
        pooled_feat_s2 = self._head_to_tail(pooled_feat_s2)

        # compute bbox offset
        bbox_pred_s1 = self.RCNN_bbox_pred(pooled_feat_s1)
        bbox_pred_s2 = self.RCNN_bbox_pred(pooled_feat_s2)

        if self.training and not self.class_agnostic:
            # select the corresponding columns according to roi labels
            # 将 [N, n_classes*4] 的 bbox regression 输出 reshape 成 [N, n_classes, 4]
            # s1
            bbox_pred_view_s1 = bbox_pred_s1.view(bbox_pred_s1.size(0), int(bbox_pred_s1.size(1) / 4), 4)
            bbox_pred_select_s1 = torch.gather(bbox_pred_view_s1, 1, rois_label_s1.view(rois_label_s1.size(0), 1, 1).expand(rois_label_s1.size(0), 1, 4))
            bbox_pred_s1 = bbox_pred_select_s1.squeeze(1)
            # s2
            bbox_pred_view_s2 = bbox_pred_s2.view(bbox_pred_s2.size(0), int(bbox_pred_s2.size(1) / 4), 4)
            bbox_pred_select_s2 = torch.gather(bbox_pred_view_s2, 1, rois_label_s2.view(rois_label_s2.size(0), 1, 1).expand(rois_label_s2.size(0), 1, 4))
            bbox_pred_s2 = bbox_pred_select_s2.squeeze(1)

        # compute object classification probability
        cls_score_s1 = self.RCNN_cls_score(pooled_feat_s1)
        cls_prob_s1 = F.softmax(cls_score_s1, 1)
        cls_score_s2 = self.RCNN_cls_score(pooled_feat_s2)
        cls_prob_s2 = F.softmax(cls_score_s2, 1)

        RCNN_loss_cls_s1 = 0
        RCNN_loss_bbox_s1 = 0
        RCNN_loss_cls_s2 = 0
        RCNN_loss_bbox_s2 = 0

        if self.training:
            # classification loss
            RCNN_loss_cls_s1 = F.cross_entropy(cls_score_s1, rois_label_s1)
            RCNN_loss_cls_s2 = F.cross_entropy(cls_score_s2, rois_label_s2)

            # bounding box regression L1 loss
            RCNN_loss_bbox_s1 = _smooth_l1_loss(bbox_pred_s1, rois_target_s1, rois_inside_ws_s1, rois_outside_ws_s1)
            RCNN_loss_bbox_s2 = _smooth_l1_loss(bbox_pred_s2, rois_target_s2, rois_inside_ws_s2, rois_outside_ws_s2)

            # ============ 【目标域伪标签生成 + RCNN监督训练】 ============
            # Step 1: 目标域图像送入 backbone + RPN    (提取目标域特征是否放在检测训练状态外？)
            base_feat_t = self.RCNN_base(im_data_t)
            rois_t, _, _ = self.RCNN_rpn(base_feat_t, im_info_t, None, None)

            # 调试：打印形状
            # print(f"rois_t shape: {rois_t.shape}")
            # print(f"base_feat_t shape: {base_feat_t.shape}")

            # 确保rois_t是正确的形状
            if rois_t.dim() == 3 and rois_t.size(0) == 1:
                rois_t = rois_t.squeeze(0)  # 移除batch维度

            # Step 2: ROI Align
            if cfg.POOLING_MODE == 'align':
                pooled_feat_t = self.RCNN_roi_align(base_feat_t, rois_t.view(-1, 5))
            else:
                pooled_feat_t = self.RCNN_roi_pool(base_feat_t, rois_t.view(-1, 5))

            # print("[DEBUG] pooled_feat_t.shape:", pooled_feat_t.shape)

            # Step 3: RCNN head 推理
            pooled_feat_t = self._head_to_tail(pooled_feat_t)
            cls_score_t = self.RCNN_cls_score(pooled_feat_t)
            cls_prob_t = F.softmax(cls_score_t, dim=1)
            bbox_pred_t = self.RCNN_bbox_pred(pooled_feat_t)

            # 调试：在pseudo_label调用前
            # print(f"Before pseudo_label - cls_prob_t shape: {cls_prob_t.shape}")
            # print(f"Before pseudo_label - bbox_pred_t shape: {bbox_pred_t.shape}")

            # Step 4: 生成伪标签（不需要)
            # 计算当前的阈值（随训练进行逐渐提高）
            with torch.no_grad():
                pseudo_boxes, pseudo_labels = pseudo_label(rois_t, cls_prob_t, bbox_pred_t, im_info_t,
                                                       score_thresh=0.5, nms_thresh=0.3,
                                                       class_agnostic=self.class_agnostic)

            # 如果无伪标签则跳过该批次
            if pseudo_boxes.size(0) == 0:
                # 训练时观察到哪些 batch 的目标图像无法生成伪标签，可能需要调整阈值（score、NMS）或改善源域性能。
                # print("[PseudoLabel] No valid target RoIs found. Skipping pseudo loss.")
                RCNN_loss_cls_t = torch.tensor(0.0).cuda()
                RCNN_loss_bbox_t = torch.tensor(0.0).cuda()
            else:
                # Step 5: 构造伪标签的 gt_boxes_t
                batch_size = im_data_t.size(0)

                # 统计每个batch的伪标签数量
                batch_counts = []
                for b in range(batch_size):
                    count = (pseudo_boxes[:, 0] == b).sum().item()
                    batch_counts.append(count)

                max_num_boxes = max(batch_counts) if batch_counts else 1

                # 创建3D的gt_boxes_t
                gt_boxes_t = torch.zeros(batch_size, max_num_boxes, 5).to(pseudo_boxes.device)
                num_boxes_t = torch.zeros(batch_size, dtype=torch.long).to(pseudo_boxes.device)

                # 填充每个batch的数据
                for b in range(batch_size):
                    mask = pseudo_boxes[:, 0] == b
                    b_boxes = pseudo_boxes[mask]
                    b_labels = pseudo_labels[mask]

                    if b_boxes.size(0) > 0:
                        num_boxes_t[b] = b_boxes.size(0)
                        gt_boxes_t[b, :b_boxes.size(0), 0:4] = b_boxes[:, 1:5]
                        gt_boxes_t[b, :b_boxes.size(0), 4] = b_labels

                # print(f"\n[DEBUG] Pseudo labels quality check:")    # 开始
                # print(f"Total pseudo boxes: {pseudo_boxes.size(0)}")

                # 检查每个batch的伪标签分布
                for b in range(batch_size):
                    mask = pseudo_boxes[:, 0] == b
                    if mask.any():
                        b_boxes = pseudo_boxes[mask]
                        b_labels = pseudo_labels[mask]
                        # print(f"  Batch {b}: {mask.sum().item()} pseudo boxes")

                        # 统计类别分布
                        unique_labels, counts = torch.unique(b_labels, return_counts=True)
                        # for lbl, cnt in zip(unique_labels, counts):
                            # print(f"    Class {lbl.item()}: {cnt.item()} boxes")

                        # 检查框的大小是否合理
                        widths = b_boxes[:, 3] - b_boxes[:, 1]
                        heights = b_boxes[:, 4] - b_boxes[:, 2]
                        areas = widths * heights
                        # print(f"    Box sizes - Min area: {areas.min().item():.1f}, "
                        #       f"Max area: {areas.max().item():.1f}, "
                        #       f"Mean area: {areas.mean().item():.1f}")

                # Step 6: Proposal target (生成 rois_target_t)
                # 注意：pooled_feat_t（原始的目标域特征）仍然要用于域鉴别器
                # 所以在域鉴别器部分，确保使用的是pooled_feat_t而不是pooled_feat_t_pseudo
                # print(f"\n[DEBUG] RPN proposals check:")        # 开始
                for b in range(batch_size):
                    b_rois = rois_t[b]
                    # 过滤掉全0的rois
                    valid_mask = (b_rois[:, 1:5].sum(dim=1) > 0)
                    valid_rois = b_rois[valid_mask]
                    # print(f"  Batch {b}: {valid_mask.sum().item()} valid RoIs out of {b_rois.size(0)}")

                    if valid_rois.size(0) > 0:
                        # 检查RoI的大小分布
                        widths = valid_rois[:, 3] - valid_rois[:, 1]
                        heights = valid_rois[:, 4] - valid_rois[:, 2]
                        areas = widths * heights
                        # print(f"    RoI areas - Min: {areas.min().item():.1f}, "
                        #       f"Max: {areas.max().item():.1f}, "
                        #       f"Mean: {areas.mean().item():.1f}")

                # 临时降低IoU阈值以便调试
                # print(f"\n[DEBUG] Original thresholds - FG: {cfg.TRAIN.FG_THRESH}, BG_HI: {cfg.TRAIN.BG_THRESH_HI}, BG_LO: {cfg.TRAIN.BG_THRESH_LO}")
                original_fg_thresh = cfg.TRAIN.FG_THRESH
                original_bg_thresh_hi = cfg.TRAIN.BG_THRESH_HI
                cfg.TRAIN.FG_THRESH = 0.3  # 从0.5降低到0.3
                cfg.TRAIN.BG_THRESH_HI = 0.3  # 同步降低    # 结束

                roi_data_t = self.RCNN_proposal_target(rois_t, gt_boxes_t, num_boxes_t)

                # 恢复原始值
                cfg.TRAIN.FG_THRESH = original_fg_thresh
                cfg.TRAIN.BG_THRESH_HI = original_bg_thresh_hi

                rois_t_pseudo, rois_label_t, rois_target_t, rois_inside_ws_t, rois_outside_ws_t = roi_data_t
                # 添加调试信息
                # print(f"[DEBUG] After proposal target:")
                # print(f"  rois_t_pseudo shape: {rois_t_pseudo.shape}")
                # print(f"  rois_label_t shape: {rois_label_t.shape}")
                # print(f"  Foreground samples: {(rois_label_t > 0).sum().item()}")
                # print(f"  Background samples: {(rois_label_t == 0).sum().item()}")

                # 分析前景样本的类别分布
                # if (rois_label_t > 0).sum() > 0:
                #     fg_labels = rois_label_t[rois_label_t > 0]
                #     unique_labels, counts = torch.unique(fg_labels, return_counts=True)
                    # print(f"  Foreground class distribution:")
                    # for lbl, cnt in zip(unique_labels, counts):
                        # print(f"    Class {lbl.item()}: {cnt.item()} samples")

                    # 检查IoU分布（需要计算伪标签与RPN proposals的IoU）
                    # print(f"\n[DEBUG] Checking IoU between pseudo labels and RPN proposals...")

                # 检查是否所有batch都没有前景样本
                if (rois_label_t > 0).sum() == 0:
                    # print("[PseudoLabel] No foreground samples after proposal target. Skipping pseudo loss.")
                    RCNN_loss_cls_t = torch.tensor(0.0).cuda()
                    RCNN_loss_bbox_t = torch.tensor(0.0).cuda()
                else:
                    rois_label_t = Variable(rois_label_t.view(-1).long())
                    rois_target_t = Variable(rois_target_t.view(-1, rois_target_t.size(2)))
                    rois_inside_ws_t = Variable(rois_inside_ws_t.view(-1, rois_inside_ws_t.size(2)))
                    rois_outside_ws_t = Variable(rois_outside_ws_t.view(-1, rois_outside_ws_t.size(2)))
                    rois_t_pseudo = Variable(rois_t_pseudo)

                # Step 7: 再次做 pooling
                    if cfg.POOLING_MODE == 'align':
                        pooled_feat_t_pseudo = self.RCNN_roi_align(base_feat_t, rois_t_pseudo.view(-1, 5))
                    else:
                        pooled_feat_t_pseudo = self.RCNN_roi_pool(base_feat_t, rois_t_pseudo.view(-1, 5))

                    pooled_feat_t_pseudo = self._head_to_tail(pooled_feat_t_pseudo)
                    cls_score_t_pseudo = self.RCNN_cls_score(pooled_feat_t_pseudo)
                    bbox_pred_t_pseudo = self.RCNN_bbox_pred(pooled_feat_t_pseudo)

                    # Step 8: 若非 class_agnostic，需要 gather 正确类 bbox
                    if not self.class_agnostic:
                        bbox_pred_view_t = bbox_pred_t_pseudo.view(bbox_pred_t_pseudo.size(0), int(bbox_pred_t_pseudo.size(1) / 4), 4)
                        bbox_pred_select_t = torch.gather(bbox_pred_view_t, 1, rois_label_t.view(-1, 1, 1).long().expand(-1, 1, 4))
                        bbox_pred_t_pseudo = bbox_pred_select_t.squeeze(1)

                    # Step 9: RCNN loss（伪监督）
                    RCNN_loss_cls_t = F.cross_entropy(cls_score_t_pseudo, rois_label_t)
                    RCNN_loss_bbox_t = _smooth_l1_loss(bbox_pred_t_pseudo, rois_target_t, rois_inside_ws_t, rois_outside_ws_t)

        cls_prob_s1 = cls_prob_s1.view(batch_size, rois_s1.size(1), -1)
        bbox_pred_s1 = bbox_pred_s1.view(batch_size, rois_s1.size(1), -1)
        cls_prob_s2 = cls_prob_s2.view(batch_size, rois_s2.size(1), -1)
        bbox_pred_s2 = bbox_pred_s2.view(batch_size, rois_s2.size(1), -1)

        # region [DA loss]
        """ start -------------------------- DA loss ---------------------------- """
        """ start ----------- image level ------------- """
        # region [image level]

        # region [img 单视角]，将原始特征直接送入 ImageDA 判别器，这是传统 DANN-style 图像级对抗训练
        base_score_s1, base_label_s1 = self.RCNN_imageDA(base_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        base_score_s2, base_label_s2 = self.RCNN_imageDA(base_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))

        base_prob_s1 = F.log_softmax(base_score_s1, dim=1)
        DA_img_loss_cls_s1 = F.nll_loss(base_prob_s1, base_label_s1)

        base_prob_s2 = F.log_softmax(base_score_s2, dim=1)
        DA_img_loss_cls_s2 = F.nll_loss(base_prob_s2, base_label_s2)

        # === [新增] 图像级 ST 域判别器（图像级是乘以batch_size） ===
        # domain_label_st_s1 = Variable(torch.LongTensor([0] * batch_size).cuda())
        # domain_label_st_t = Variable(torch.LongTensor([1] * batch_size).cuda())
        #
        # score_st_s1 = self.RCNN_imageDA_ST(base_feat_s1)
        # score_st_t = self.RCNN_imageDA_ST(base_feat_t)
        #
        # DA_img_loss_cls_ST_s1 = self.da_img_loss_fn(score_st_s1, domain_label_st_s1)
        # DA_img_loss_cls_ST_t = self.da_img_loss_fn(score_st_t, domain_label_st_t)
        #
        # DA_img_loss_cls_ST = DA_img_loss_cls_ST_s1 + DA_img_loss_cls_ST_t
        base_score_st_s1, base_label_st_s1 = self.RCNN_imageDA_ST(base_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        base_score_st_t, base_label_st_t = self.RCNN_imageDA_ST(base_feat_t, Variable(torch.FloatTensor([1.] * batch_size).cuda()))

        base_prob_st_s1 = F.log_softmax(base_score_st_s1, dim=1)
        DA_img_loss_cls_ST_s1 = F.nll_loss(base_prob_st_s1, base_label_st_s1)

        base_prob_st_t = F.log_softmax(base_score_st_t, dim=1)
        DA_img_loss_cls_ST_t = F.nll_loss(base_prob_st_t, base_label_st_t)

        DA_img_loss_cls_ST = DA_img_loss_cls_ST_s1 + DA_img_loss_cls_ST_t

        # endregion

        # region [img MV]

        # 上采样
        base_feat_s1_re = self.upsample(base_feat_s1)
        base_feat_s2_re = self.upsample(base_feat_s2)
        base_feat_t_re = self.upsample(base_feat_t)  # 策略A：加入目标域

        # region [视角1]
        MV1_feat_s1 = self.AutoEn_1(base_feat_s1_re)
        MV1_feat_s2 = self.AutoEn_1(base_feat_s2_re)
        MV1_feat_t = self.AutoEn_1(base_feat_t_re)  # 策略A：加入目标域
        re1_feat_s1 = self.AutoDe_1(MV1_feat_s1)
        re1_feat_s2 = self.AutoDe_1(MV1_feat_s2)
        re1_feat_t = self.AutoDe_1(MV1_feat_t)  # 策略A：加入目标域
        # 归一化
        MV1_feat_s1 = self.ln_img(MV1_feat_s1)
        MV1_feat_s2 = self.ln_img(MV1_feat_s2)
        MV1_feat_t = self.ln_img(MV1_feat_t)  # 策略A：加入目标域
        # 判别器损失
        MV1_score_s1, MV1_label_s1 = self.RCNN_imageDA_en1(MV1_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        MV1_score_s2, MV1_label_s2 = self.RCNN_imageDA_en1(MV1_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))

        MV1_prob_s1_en = F.log_softmax(MV1_score_s1, dim=1)
        DA_img_MV1_s1 = F.nll_loss(MV1_prob_s1_en, MV1_label_s1)

        MV1_prob_s2_en = F.log_softmax(MV1_score_s2, dim=1)
        DA_img_MV1_s2 = F.nll_loss(MV1_prob_s2_en, MV1_label_s2)

        # 策略A：S1 vs T 判别器（需要新增 self.RCNN_imageDA_en1_ST）
        MV1_score_s1_vs_t, MV1_label_s1_vs_t = self.RCNN_imageDA_en1_ST(MV1_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        MV1_score_t_vs_s1, MV1_label_t_vs_s1 = self.RCNN_imageDA_en1_ST(MV1_feat_t, Variable(torch.FloatTensor([1.] * batch_size).cuda()))

        MV1_prob_s1_vs_t = F.log_softmax(MV1_score_s1_vs_t, dim=1)
        DA_img_MV1_s1_vs_t = F.nll_loss(MV1_prob_s1_vs_t, MV1_label_s1_vs_t)

        MV1_prob_t_vs_s1 = F.log_softmax(MV1_score_t_vs_s1, dim=1)
        DA_img_MV1_t_vs_s1 = F.nll_loss(MV1_prob_t_vs_s1, MV1_label_t_vs_s1)

        # recon_loss（重构损失）
        MV1_s1 = self.mse_loss(re1_feat_s1, base_feat_s1_re.detach())
        MV1_s2 = self.mse_loss(re1_feat_s2, base_feat_s2_re.detach())
        MV1_t = self.mse_loss(re1_feat_t, base_feat_t_re.detach())  # 策略A：加入目标域

        # 重构损失 + 域分类损失
        # img_MV1_loss = MV1_s1 + MV1_s2 + DA_img_MV1_s1 + DA_img_MV1_s2
        img_MV1_recon_loss = MV1_s1 + MV1_s2 + MV1_t
        img_MV1_cls_loss = DA_img_MV1_s1 + DA_img_MV1_s2 + DA_img_MV1_s1_vs_t + DA_img_MV1_t_vs_s1

        # endregion

        # region [视角2]
        MV2_feat_s1 = self.AutoEn_2(base_feat_s1_re)
        MV2_feat_s2 = self.AutoEn_2(base_feat_s2_re)
        MV2_feat_t = self.AutoEn_2(base_feat_t_re)  # 策略A：加入目标域
        re2_feat_s1 = self.AutoDe_2(MV2_feat_s1)
        re2_feat_s2 = self.AutoDe_2(MV2_feat_s2)
        re2_feat_t = self.AutoDe_2(MV2_feat_t)  # 策略A：加入目标域
        # 归一化
        MV2_feat_s1 = self.ln_img(MV2_feat_s1)
        MV2_feat_s2 = self.ln_img(MV2_feat_s2)
        MV2_feat_t = self.ln_img(MV2_feat_t)  # 策略A：加入目标域
        # 判别器损失
        MV2_score_s1, MV2_label_s1_en = self.RCNN_imageDA_en2(MV2_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        MV2_score_s2, MV2_label_s2_en = self.RCNN_imageDA_en2(MV2_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))

        MV2_prob_s1_en = F.log_softmax(MV2_score_s1, dim=1)
        DA_img_MV2_s1 = F.nll_loss(MV2_prob_s1_en, MV2_label_s1_en)

        MV2_prob_s2_en = F.log_softmax(MV2_score_s2, dim=1)
        DA_img_MV2_s2 = F.nll_loss(MV2_prob_s2_en, MV2_label_s2_en)

        # 策略A：S1 vs T 判别器
        MV2_score_s1_vs_t, MV2_label_s1_vs_t = self.RCNN_imageDA_en2_ST(MV2_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        MV2_score_t_vs_s1, MV2_label_t_vs_s1 = self.RCNN_imageDA_en2_ST(MV2_feat_t, Variable(torch.FloatTensor([1.] * batch_size).cuda()))

        MV2_prob_s1_vs_t = F.log_softmax(MV2_score_s1_vs_t, dim=1)
        DA_img_MV2_s1_vs_t = F.nll_loss(MV2_prob_s1_vs_t, MV2_label_s1_vs_t)

        MV2_prob_t_vs_s1 = F.log_softmax(MV2_score_t_vs_s1, dim=1)
        DA_img_MV2_t_vs_s1 = F.nll_loss(MV2_prob_t_vs_s1, MV2_label_t_vs_s1)

        # recon_loss
        MV2_s1 = self.mse_loss(re2_feat_s1, base_feat_s1_re.detach())
        MV2_s2 = self.mse_loss(re2_feat_s2, base_feat_s2_re.detach())
        MV2_t = self.mse_loss(re2_feat_t, base_feat_t_re.detach())  # 策略A：加入目标域

        # 重构损失 + 域分类损失
        # img_MV2_loss = MV2_s1 + MV2_s2 + DA_img_MV2_s1 + DA_img_MV2_s2
        img_MV2_recon_loss = MV2_s1 + MV2_s2 + MV2_t
        img_MV2_cls_loss = DA_img_MV2_s1 + DA_img_MV2_s2 + DA_img_MV2_s1_vs_t + DA_img_MV2_t_vs_s1

        # endregion

        # region [视角3]
        MV3_feat_s1 = self.AutoEn_3(base_feat_s1_re)
        MV3_feat_s2 = self.AutoEn_3(base_feat_s2_re)
        MV3_feat_t = self.AutoEn_3(base_feat_t_re)  # 策略A：加入目标域
        re3_feat_s1 = self.AutoDe_3(MV3_feat_s1)
        re3_feat_s2 = self.AutoDe_3(MV3_feat_s2)
        re3_feat_t = self.AutoDe_3(MV3_feat_t)  # 策略A：加入目标域
        # 归一化
        MV3_feat_s1 = self.ln_img(MV3_feat_s1)
        MV3_feat_s2 = self.ln_img(MV3_feat_s2)
        MV3_feat_t = self.ln_img(MV3_feat_t)  # 策略A：加入目标域
        # 判别器损失
        MV3_score_s1, MV3_label_s1_en = self.RCNN_imageDA_en3(MV3_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        MV3_score_s2, MV3_label_s2_en = self.RCNN_imageDA_en3(MV3_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))

        MV3_prob_s1_en = F.log_softmax(MV3_score_s1, dim=1)
        DA_img_MV3_s1 = F.nll_loss(MV3_prob_s1_en, MV3_label_s1_en)

        MV3_prob_s2_en = F.log_softmax(MV3_score_s2, dim=1)
        DA_img_MV3_s2 = F.nll_loss(MV3_prob_s2_en, MV3_label_s2_en)

        # 策略A：S1 vs T 判别器
        MV3_score_s1_vs_t, MV3_label_s1_vs_t = self.RCNN_imageDA_en3_ST(MV3_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        MV3_score_t_vs_s1, MV3_label_t_vs_s1 = self.RCNN_imageDA_en3_ST(MV3_feat_t, Variable(torch.FloatTensor([1.] * batch_size).cuda()))

        MV3_prob_s1_vs_t = F.log_softmax(MV3_score_s1_vs_t, dim=1)
        DA_img_MV3_s1_vs_t = F.nll_loss(MV3_prob_s1_vs_t, MV3_label_s1_vs_t)

        MV3_prob_t_vs_s1 = F.log_softmax(MV3_score_t_vs_s1, dim=1)
        DA_img_MV3_t_vs_s1 = F.nll_loss(MV3_prob_t_vs_s1, MV3_label_t_vs_s1)

        # recon_loss
        MV3_s1 = self.mse_loss(re3_feat_s1, base_feat_s1_re.detach())
        MV3_s2 = self.mse_loss(re3_feat_s2, base_feat_s2_re.detach())
        MV3_t = self.mse_loss(re3_feat_t, base_feat_t_re.detach())  # 策略A：加入目标域

        # 重构损失 + 域分类损失
        img_MV3_recon_loss = MV3_s1 + MV3_s2 + MV3_t
        img_MV3_cls_loss = DA_img_MV3_s1 + DA_img_MV3_s2 + DA_img_MV3_s1_vs_t + DA_img_MV3_t_vs_s1

        # endregion
        # 计算不同视角提取到的表示之间的欧式差异
        dif12_s1 = (self.mse_loss(MV1_feat_s1, MV2_feat_s1.detach()) + self.mse_loss(MV2_feat_s1, MV1_feat_s1.detach())) / 2
        dif12_s2 = (self.mse_loss(MV1_feat_s2, MV2_feat_s2.detach()) + self.mse_loss(MV2_feat_s2, MV1_feat_s2.detach())) / 2
        dif12_t = (self.mse_loss(MV1_feat_t, MV2_feat_t.detach()) + self.mse_loss(MV2_feat_t, MV1_feat_t.detach())) / 2

        dif13_s1 = (self.mse_loss(MV1_feat_s1, MV3_feat_s1.detach()) + self.mse_loss(MV3_feat_s1, MV1_feat_s1.detach())) / 2
        dif13_s2 = (self.mse_loss(MV1_feat_s2, MV3_feat_s2.detach()) + self.mse_loss(MV3_feat_s2, MV1_feat_s2.detach())) / 2
        dif13_t = (self.mse_loss(MV1_feat_t, MV3_feat_t.detach()) + self.mse_loss(MV3_feat_t, MV1_feat_t.detach())) / 2

        dif23_s1 = (self.mse_loss(MV3_feat_s1, MV2_feat_s1.detach()) + self.mse_loss(MV2_feat_s1, MV3_feat_s1.detach())) / 2
        dif23_s2 = (self.mse_loss(MV3_feat_s2, MV2_feat_s2.detach()) + self.mse_loss(MV2_feat_s2, MV3_feat_s2.detach())) / 2
        dif23_t = (self.mse_loss(MV3_feat_t, MV2_feat_t.detach()) + self.mse_loss(MV2_feat_t, MV3_feat_t.detach())) / 2

        # 再通过 exp(-x) 函数压缩成 “多样性约束值”
        # 表示“视角间越不一样越好”,以防止模型所有视角提取出的是同质化特征
        # 增强多视角系统的互补性而非冗余
        img_mv_dis_loss = torch.exp(-(dif12_s1 + dif12_s2 + dif12_t + dif13_s1 + dif13_s2 + dif13_t + dif23_s1 + dif23_s2 + dif23_t))
        # img_MV_loss = img_MV1_loss + img_MV2_loss + img_MV3_loss + un_dis # - (0.01) * (dif_s1 + dif_s2)
        # 多视角重建损失 + 域判别损失
        img_mv_recon_loss = img_MV3_recon_loss + img_MV2_recon_loss + img_MV1_recon_loss
        img_mv_cls_loss = img_MV3_cls_loss + img_MV2_cls_loss + img_MV1_cls_loss

        # endregion

        # endregion

        """ start ----------- instance level ------------- """
        # region [instance level]

        # region [ins 单视角]
        DA_ins_loss_cls_s1 = 0
        DA_ins_loss_cls_s2 = 0

        # pooled_feat_s1 是每个 RoI 的全连接特征，判断 RoI 是否来自源域/目标域，输出 sigmoid 形式。损失函数：使用 Binary Cross Entropy 与 GT 域标签对比
        # s1
        instance_sigmoid_s1, same_size_label_s1 = self.RCNN_instanceDA(pooled_feat_s1, domain_label_s1)
        instance_loss_s1 = nn.BCELoss()
        DA_ins_loss_cls_s1 = instance_loss_s1(instance_sigmoid_s1, same_size_label_s1)
        # s2
        instance_sigmoid_s2, same_size_label_s2 = self.RCNN_instanceDA(pooled_feat_s2, domain_label_s2)
        instance_loss_s2 = nn.BCELoss()
        DA_ins_loss_cls_s2 = instance_loss_s2(instance_sigmoid_s2, same_size_label_s2)

        # === [新增] 实例级 ST 域判别器（实例级要乘以pooled_feat，但是DA.py中使用了InstanceLabelResizeLayer，所以能够使用batch_size） ===
        domain_label_st_ins_s1 = Variable(torch.FloatTensor([0.] * batch_size).cuda())
        domain_label_st_ins_t = Variable(torch.FloatTensor([1.] * batch_size).cuda())

        sigmoid_st_s1, label_st_s1 = self.RCNN_instanceDA_ST(pooled_feat_s1, domain_label_st_ins_s1)
        sigmoid_st_t, label_st_t = self.RCNN_instanceDA_ST(pooled_feat_t, domain_label_st_ins_t)

        DA_ins_loss_cls_ST_s1 = self.da_ins_loss_fn(sigmoid_st_s1, label_st_s1)
        DA_ins_loss_cls_ST_t = self.da_ins_loss_fn(sigmoid_st_t, label_st_t)

        DA_ins_loss_cls_ST = DA_ins_loss_cls_ST_s1 + DA_ins_loss_cls_ST_t

        # endregion

        # region [ins MV]

        # region [视角1]
        # 特征扰动与压缩
        MV1_ins_feat_s1 = self.InsEn_1(pooled_feat_s1)
        MV1_ins_feat_s2 = self.InsEn_1(pooled_feat_s2)
        MV1_ins_feat_t = self.InsEn_1(pooled_feat_t)  # 策略A：加入目标域
        # 特征重建
        re1_ins_feat_s1 = self.InsDe_1(MV1_ins_feat_s1)
        re1_ins_feat_s2 = self.InsDe_1(MV1_ins_feat_s2)
        re1_ins_feat_t = self.InsDe_1(MV1_ins_feat_t)  # 策略A：加入目标域
        # 归一化
        MV1_ins_feat_s1 = self.ln_ins(MV1_ins_feat_s1)
        MV1_ins_feat_s2 = self.ln_ins(MV1_ins_feat_s2)
        MV1_ins_feat_t = self.ln_ins(MV1_ins_feat_t)  # 策略A：加入目标域

        # 1. recon_loss(重构损失)
        MV1_ins_s1 = self.mse_loss(re1_ins_feat_s1, pooled_feat_s1.detach())
        MV1_ins_s2 = self.mse_loss(re1_ins_feat_s2, pooled_feat_s2.detach())
        MV1_ins_t = self.mse_loss(re1_ins_feat_t, pooled_feat_t.detach())  # 策略A：加入目标域

        # 2. 实例域分类损失
        # s1
        instance_sigmoid_s1_MV1, same_size_label_s1_MV1 = self.RCNN_insDA_en1(MV1_ins_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        DA_ins_MV1_s1 = self.RCNN_insDA_loss(instance_sigmoid_s1_MV1, same_size_label_s1_MV1)
        # s2
        instance_sigmoid_s2_MV1, same_size_label_s2_MV1 = self.RCNN_insDA_en1(MV1_ins_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        DA_ins_MV1_s2 = self.RCNN_insDA_loss(instance_sigmoid_s2_MV1, same_size_label_s2_MV1)

        # 策略A：S1 vs T（需要新增 self.RCNN_insDA_en1_ST）
        instance_sigmoid_s1_MV1_ST, same_size_label_s1_MV1_ST = self.RCNN_insDA_en1_ST(MV1_ins_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        DA_ins_MV1_s1_vs_t = self.RCNN_insDA_loss(instance_sigmoid_s1_MV1_ST, same_size_label_s1_MV1_ST)

        instance_sigmoid_t_MV1_ST, same_size_label_t_MV1_ST = self.RCNN_insDA_en1_ST(MV1_ins_feat_t, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        DA_ins_MV1_t_vs_s1 = self.RCNN_insDA_loss(instance_sigmoid_t_MV1_ST, same_size_label_t_MV1_ST)

        # ins_MV1_loss = DA_ins_MV1_s1 + DA_ins_MV1_s2 + MV1_ins_s1 + MV1_ins_s2
        # ins_MVx_recon_loss: 重构损失，保持语义稳定        ins_MVx_cls_loss: 对抗损失，压制域相关性
        ins_MV1_recon_loss = MV1_ins_s1 + MV1_ins_s2 + MV1_ins_t
        ins_MV1_cls_loss = DA_ins_MV1_s1 + DA_ins_MV1_s2 + DA_ins_MV1_s1_vs_t + DA_ins_MV1_t_vs_s1

        # endregion

        # region [视角2]
        MV2_ins_feat_s1 = self.InsEn_2(pooled_feat_s1)
        MV2_ins_feat_s2 = self.InsEn_2(pooled_feat_s2)
        MV2_ins_feat_t = self.InsEn_2(pooled_feat_t)  # 策略A：加入目标域
        re2_ins_feat_s1 = self.InsDe_2(MV2_ins_feat_s1)
        re2_ins_feat_s2 = self.InsDe_2(MV2_ins_feat_s2)
        re2_ins_feat_t = self.InsDe_2(MV2_ins_feat_t)  # 策略A：加入目标域
        # 归一化
        MV2_ins_feat_s1 = self.ln_ins(MV2_ins_feat_s1)
        MV2_ins_feat_s2 = self.ln_ins(MV2_ins_feat_s2)
        MV2_ins_feat_t = self.ln_ins(MV2_ins_feat_t)  # 策略A：加入目标域

        # 1. recon_loss(重构损失)
        MV2_ins_s1 = self.mse_loss(re2_ins_feat_s1, pooled_feat_s1.detach())
        MV2_ins_s2 = self.mse_loss(re2_ins_feat_s2, pooled_feat_s2.detach())
        MV2_ins_t = self.mse_loss(re2_ins_feat_t, pooled_feat_t.detach())  # 策略A：加入目标域

        # 2. 领域分类损失
        # s1
        instance_sigmoid_s1_MV2, same_size_label_s1_MV2 = self.RCNN_insDA_en2(MV2_ins_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        DA_ins_MV2_s1 = self.RCNN_insDA_loss(instance_sigmoid_s1_MV2, same_size_label_s1_MV2)
        # s2
        instance_sigmoid_s2_MV2, same_size_label_s2_MV2 = self.RCNN_insDA_en2(MV2_ins_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        DA_ins_MV2_s2 = self.RCNN_insDA_loss(instance_sigmoid_s2_MV2, same_size_label_s2_MV2)

        # 策略A：S1 vs T
        instance_sigmoid_s1_MV2_ST, same_size_label_s1_MV2_ST = self.RCNN_insDA_en2_ST(MV2_ins_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        DA_ins_MV2_s1_vs_t = self.RCNN_insDA_loss(instance_sigmoid_s1_MV2_ST, same_size_label_s1_MV2_ST)

        instance_sigmoid_t_MV2_ST, same_size_label_t_MV2_ST = self.RCNN_insDA_en2_ST(MV2_ins_feat_t, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        DA_ins_MV2_t_vs_s1 = self.RCNN_insDA_loss(instance_sigmoid_t_MV2_ST, same_size_label_t_MV2_ST)

        # ins_MV2_loss = DA_ins_MV2_s1 + DA_ins_MV2_s2 + MV2_ins_s1 + MV2_ins_s2
        ins_MV2_recon_loss = MV2_ins_s1 + MV2_ins_s2 + MV2_ins_t
        ins_MV2_cls_loss = DA_ins_MV2_s1 + DA_ins_MV2_s2 + DA_ins_MV2_s1_vs_t + DA_ins_MV2_t_vs_s1

        # endregion

        # region [视角3]
        MV3_ins_feat_s1 = self.InsEn_3(pooled_feat_s1)
        MV3_ins_feat_s2 = self.InsEn_3(pooled_feat_s2)
        MV3_ins_feat_t = self.InsEn_3(pooled_feat_t)  # 策略A：加入目标域
        re3_ins_feat_s1 = self.InsDe_3(MV3_ins_feat_s1)
        re3_ins_feat_s2 = self.InsDe_3(MV3_ins_feat_s2)
        re3_ins_feat_t = self.InsDe_3(MV3_ins_feat_t)  # 策略A：加入目标域
        # 归一化
        MV3_ins_feat_s1 = self.ln_ins(MV3_ins_feat_s1)
        MV3_ins_feat_s2 = self.ln_ins(MV3_ins_feat_s2)
        MV3_ins_feat_t = self.ln_ins(MV3_ins_feat_t)  # 策略A：加入目标域

        # 1. recon_loss(重构损失)
        MV3_ins_s1 = self.mse_loss(re3_ins_feat_s1, pooled_feat_s1.detach())
        MV3_ins_s2 = self.mse_loss(re3_ins_feat_s2, pooled_feat_s2.detach())
        MV3_ins_t = self.mse_loss(re3_ins_feat_t, pooled_feat_t.detach())  # 策略A：加入目标域

        # 2. 领域分类损失
        # s1
        instance_sigmoid_s1_MV3, same_size_label_s1_MV3 = self.RCNN_insDA_en3(MV3_ins_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        DA_ins_MV3_s1 = self.RCNN_insDA_loss(instance_sigmoid_s1_MV3, same_size_label_s1_MV3)
        # s2
        instance_sigmoid_s2_MV3, same_size_label_s2_MV3 = self.RCNN_insDA_en3(MV3_ins_feat_s2, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        DA_ins_MV3_s2 = self.RCNN_insDA_loss(instance_sigmoid_s2_MV3, same_size_label_s2_MV3)

        instance_sigmoid_s1_MV3_ST, same_size_label_s1_MV3_ST = self.RCNN_insDA_en3_ST(MV3_ins_feat_s1, Variable(torch.FloatTensor([0.] * batch_size).cuda()))
        DA_ins_MV3_s1_vs_t = self.RCNN_insDA_loss(instance_sigmoid_s1_MV3_ST, same_size_label_s1_MV3_ST)

        instance_sigmoid_t_MV3_ST, same_size_label_t_MV3_ST = self.RCNN_insDA_en3_ST(MV3_ins_feat_t, Variable(torch.FloatTensor([1.] * batch_size).cuda()))
        DA_ins_MV3_t_vs_s1 = self.RCNN_insDA_loss(instance_sigmoid_t_MV3_ST, same_size_label_t_MV3_ST)

        # ins_MV3_loss = DA_ins_MV3_s1 + DA_ins_MV3_s2 + MV3_ins_s1 + MV3_ins_s2
        ins_MV3_recon_loss = MV3_ins_s1 + MV3_ins_s2 + MV3_ins_t
        ins_MV3_cls_loss = DA_ins_MV3_s1 + DA_ins_MV3_s2 + DA_ins_MV3_s1_vs_t + DA_ins_MV3_t_vs_s1

        # endregion

        # 3. 实例多视角损失
        # 意图：不同视角下的特征应具备足够“表现力差异”
        dif12_ins_s1 = (self.mse_loss(MV1_ins_feat_s1, MV2_ins_feat_s1.detach()) + self.mse_loss(MV2_ins_feat_s1, MV1_ins_feat_s1.detach())) / 2
        dif12_ins_s2 = (self.mse_loss(MV1_ins_feat_s2, MV2_ins_feat_s2.detach()) + self.mse_loss(MV2_ins_feat_s2, MV1_ins_feat_s2.detach())) / 2
        dif12_ins_t = (self.mse_loss(MV1_ins_feat_t, MV2_ins_feat_t.detach()) + self.mse_loss(MV2_ins_feat_t, MV1_ins_feat_t.detach())) / 2

        dif13_ins_s1 = (self.mse_loss(MV1_ins_feat_s1, MV3_ins_feat_s1.detach()) + self.mse_loss(MV3_ins_feat_s1, MV1_ins_feat_s1.detach())) / 2
        dif13_ins_s2 = (self.mse_loss(MV1_ins_feat_s2, MV3_ins_feat_s2.detach()) + self.mse_loss(MV3_ins_feat_s2, MV1_ins_feat_s2.detach())) / 2
        dif13_ins_t = (self.mse_loss(MV1_ins_feat_t, MV3_ins_feat_t.detach()) + self.mse_loss(MV3_ins_feat_t, MV1_ins_feat_t.detach())) / 2

        dif23_ins_s1 = (self.mse_loss(MV3_ins_feat_s1, MV2_ins_feat_s1.detach()) + self.mse_loss(MV2_ins_feat_s1, MV3_ins_feat_s1.detach())) / 2
        dif23_ins_s2 = (self.mse_loss(MV3_ins_feat_s2, MV2_ins_feat_s2.detach()) + self.mse_loss(MV2_ins_feat_s2, MV3_ins_feat_s2.detach())) / 2
        dif23_ins_t = (self.mse_loss(MV3_ins_feat_t, MV2_ins_feat_t.detach()) + self.mse_loss(MV2_ins_feat_t, MV3_ins_feat_t.detach())) / 2

        # ins_mv_dis_loss = torch.exp(-(dif12_ins_s1 + dif12_ins_s2 + dif13_ins_s1 + dif13_ins_s2 + dif23_ins_s1 + dif23_ins_s2))
        # 使用 1 / (diff_sum) 实现差异增强（差异越大，损失越小），用于防止所有 Encoder 提取出冗余信息 → 强化多样性
        ins_mv_dis_loss = 1 / (dif12_ins_s1 + dif12_ins_s2 + dif12_ins_t + dif13_ins_s1 + dif13_ins_s2 + dif13_ins_t + dif23_ins_s1 + dif23_ins_s2 + dif23_ins_t)

        # ins_MV_loss = ins_MV1_loss + ins_MV2_loss + ins_MV3_loss + un_ins_dis # - (0.01) * (dif_ins_s1 + dif_ins_s2)
        ins_mv_recon_loss = ins_MV3_recon_loss + ins_MV2_recon_loss + ins_MV1_recon_loss
        ins_mv_cls_loss = ins_MV3_cls_loss + ins_MV2_cls_loss + ins_MV1_cls_loss

        # endregion

        # endregion

        """ start ----------- consistency ------------- """
        # region [consistency level]    跨视图一致性损失
        # 意图：图像级与实例级 domain 判别结果应保持一致
        # 方法：将图像判别 softmax 概率拉平，与实例 sigmoid 输出进行均值对齐
        consistency_prob_s1 = F.softmax(base_score_s1, dim=1)[:, 1, :, :]
        consistency_prob_s1 = torch.mean(consistency_prob_s1)
        consistency_prob_s1 = consistency_prob_s1.repeat(instance_sigmoid_s1.size())
        DA_cst_loss_s1 = self.consistency_loss(instance_sigmoid_s1, consistency_prob_s1.detach())

        consistency_prob_s2 = F.softmax(base_score_s2, dim=1)[:, 0, :, :]
        consistency_prob_s2 = torch.mean(consistency_prob_s2)
        consistency_prob_s2 = consistency_prob_s2.repeat(instance_sigmoid_s2.size())
        DA_cst_loss_s2 = self.consistency_loss(instance_sigmoid_s2, consistency_prob_s2.detach())

        # soft cst
        # 意图：多视角判别器的平均结果也应一致
        # 作用：稳健输出，增强泛化能力
        MV_score_s1 = (MV1_score_s1 + MV2_score_s1 + MV3_score_s1) / 3
        MV_score_s2 = (MV1_score_s2 + MV2_score_s2 + MV3_score_s2) / 3

        instance_sigmoid_s1_MV = (instance_sigmoid_s1_MV1 + instance_sigmoid_s1_MV2 + instance_sigmoid_s1_MV3) / 3  # 实例判别器输出
        instance_sigmoid_s2_MV = (instance_sigmoid_s2_MV1 + instance_sigmoid_s2_MV2 + instance_sigmoid_s2_MV3) / 3

        soft_cst_prob_s1 = F.softmax(MV_score_s1, dim=1)[:, 1, :, :]
        soft_cst_prob_s1 = torch.mean(soft_cst_prob_s1)
        soft_cst_prob_s1 = soft_cst_prob_s1.repeat(instance_sigmoid_s1_MV.size())
        MV_cst_loss_s1 = self.consistency_loss(instance_sigmoid_s1_MV, soft_cst_prob_s1.detach())

        soft_cst_s2 = F.softmax(MV_score_s2, dim=1)[:, 0, :, :]
        soft_cst_s2 = torch.mean(soft_cst_s2)
        soft_cst_s2 = soft_cst_s2.repeat(instance_sigmoid_s2_MV.size())
        MV_cst_loss_s2 = self.consistency_loss(instance_sigmoid_s2_MV, soft_cst_s2.detach())

        # endregion

        # endregion
        return rois_s1, cls_prob_s1, bbox_pred_s1, rpn_loss_cls_s1, rpn_loss_bbox_s1, RCNN_loss_cls_s1, RCNN_loss_bbox_s1, rois_label_s1, \
            rois_s2, cls_prob_s2, bbox_pred_s2, rpn_loss_cls_s2, rpn_loss_bbox_s2, RCNN_loss_cls_s2, RCNN_loss_bbox_s2, rois_label_s2, \
            DA_img_loss_cls_s1, DA_img_loss_cls_s2, DA_ins_loss_cls_s1, DA_ins_loss_cls_s2, DA_cst_loss_s1, DA_cst_loss_s2, \
            img_mv_recon_loss, img_mv_cls_loss, img_mv_dis_loss, \
            ins_mv_recon_loss, ins_mv_cls_loss, ins_mv_dis_loss, \
            MV_cst_loss_s1, MV_cst_loss_s2, \
            DA_img_loss_cls_ST, DA_ins_loss_cls_ST, \
            RCNN_loss_cls_t, RCNN_loss_bbox_t
        # DA_img_loss_cls_ST 和 DA_ins_loss_cls_ST 属于域判别损失，RCNN_loss_cls_t 和 RCNN_loss_bbox_t 属于伪标签监督损失。

    def _init_weights(self):
        def normal_init(m, mean, stddev, truncated=False):
            """
            weight initalizer: truncated normal and random normal.
            """
            # x is a parameter
            if truncated:
                m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
            else:
                m.weight.data.normal_(mean, stddev)
                m.bias.data.zero_()

        normal_init(self.RCNN_rpn.RPN_Conv, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_rpn.RPN_bbox_pred, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_cls_score, 0, 0.01, cfg.TRAIN.TRUNCATED)
        normal_init(self.RCNN_bbox_pred, 0, 0.001, cfg.TRAIN.TRUNCATED)

    def create_architecture(self):
        self._init_modules()
        self._init_weights()
