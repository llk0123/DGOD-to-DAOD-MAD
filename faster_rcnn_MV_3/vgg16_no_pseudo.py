# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torchvision.models as models
# from model.faster_rcnn_MV_3.faster_rcnn_no_pseudo2 import _fasterRCNN
from model.faster_rcnn_MV_3.faster_rcnn_no_pseudo5 import _fasterRCNN
import pdb


class vgg16(_fasterRCNN):
    def __init__(self, classes, pretrained=False, class_agnostic=False):

        self.model_path = '../data/pretrained_model/vgg16_caffe.pth'
        self.dout_base_model = 512
        self.pretrained = pretrained
        self.class_agnostic = class_agnostic

        _fasterRCNN.__init__(self, classes, class_agnostic)

    def _init_modules(self):
        vgg = models.vgg16()
        if self.pretrained:
            print("Loading pretrained weights from %s" % (self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k: v for k, v in state_dict.items() if k in vgg.state_dict()})

        # 这里去掉最后一层 Linear(4096→1000)
        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        # 这里去掉了最后一个最大池化层（maxpool5），用于提升 RoI 特征分辨率（即保持更大的 RoI）
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        # 固定 conv1 和 conv2 层（前 10 层），避免过拟合，保留低层通用特征
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        # not using the last maxpool layer
        # 分类器输出：每个 RoI 的类别分布
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        # 回归器输出：
        # 类别相关：每类对应 4 个坐标 → 总共 4×C
        # 类别无关：只预测 4 个 bbox 坐标，适用于 class-agnostic 的检测器
        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    # 从 Pooling 特征 → FC，Pooling 特征是 (B×R, C, H, W) 的形状，需要展平为 [B×R, C×H×W]
    def _head_to_tail(self, pool5):

        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7

