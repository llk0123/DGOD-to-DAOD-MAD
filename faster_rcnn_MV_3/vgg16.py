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
from model.faster_rcnn_MV_3.faster_rcnn import _fasterRCNN
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
            print("Loading pretrained weights from %s" %(self.model_path))
            state_dict = torch.load(self.model_path)
            vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
            # print("Loading pretrained weights from", self.model_path)
            # state_dict = torch.load(self.model_path)
            # # --- 方法 1：查看匹配情况 ---
            # missing, unexpected = [], []
            # for k in vgg.state_dict().keys():
            #     if k not in state_dict:
            #         missing.append(k)
            # for k in state_dict.keys():
            #     if k not in vgg.state_dict():
            #         unexpected.append(k)
            # print(f">>> [DEBUG] missing keys: {len(missing)}  unexpected keys: {len(unexpected)}")
            # # 你可以打印部分列: print(missing[:5], unexpected[:5])
            #
            # # --- 方法 2：观察第一层是否被赋值 ---
            # w_before = vgg.features[0].weight[0, 0, :2, :2].clone()
            # # 只加载匹配的 key
            # own = vgg.state_dict()
            # for k, v in state_dict.items():
            #     if k in own:
            #         own[k].copy_(v)
            # w_after = vgg.features[0].weight[0, 0, :2, :2]
            # print(">>> [DEBUG] first-layer weight before:", w_before)
            # print(">>> [DEBUG] first-layer weight after: ", w_after)
            # # 推荐保持 strict=False 以避免加载失败
            # # vgg.load_state_dict(state_dict, strict=False)

        vgg.classifier = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])

        # not using the last maxpool layer
        self.RCNN_base = nn.Sequential(*list(vgg.features._modules.values())[:-1])

        # Fix the layers before conv3:
        for layer in range(10):
            for p in self.RCNN_base[layer].parameters(): p.requires_grad = False

        # self.RCNN_base = _RCNN_base(vgg.features, self.classes, self.dout_base_model)

        self.RCNN_top = vgg.classifier

        # not using the last maxpool layer
        self.RCNN_cls_score = nn.Linear(4096, self.n_classes)

        if self.class_agnostic:
            self.RCNN_bbox_pred = nn.Linear(4096, 4)
        else:
            self.RCNN_bbox_pred = nn.Linear(4096, 4 * self.n_classes)

    def _head_to_tail(self, pool5):
        # print(f'***** [DEBUG] pool5.shape:{pool5.shape} *****')       # 观察用
        # print(">>> [DEBUG] _head_to_tail input (pool5.shape):", pool5.shape)
        pool5_flat = pool5.view(pool5.size(0), -1)
        fc7 = self.RCNN_top(pool5_flat)

        return fc7
