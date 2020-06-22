# -*- coding: utf-8 -*-

import numpy as np
import os
import cfgs
import cv2
import math
from utils.utils import image_preprocess
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.data_aug import random_horizontal_flip, random_crop, random_translate, random_color_distort


def process_data(line, use_aug):
    if 'str' not in str(type(line)):
        line = line.decode()
    s = line.split()
    image_path = s[0]
    if not os.path.exists(image_path):
        raise KeyError("%s does not exist ... " % image_path)
    image = np.array(cv2.imread(image_path))
    labels = np.array([list(map(lambda x: int(float(x)), box.split(','))) for box in s[1:]])

    if use_aug:
        # image, labels = random_horizontal_flip(image, labels)
        image, labels = random_crop(image, labels)
        image, labels = random_translate(image, labels)
        # image = random_color_distort(image)
    image, labels = image_preprocess(np.copy(image), [cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W], np.copy(labels))

    output_h = cfgs.INPUT_IMAGE_H // cfgs.DOWN_RATIO
    output_w = cfgs.INPUT_IMAGE_W // cfgs.DOWN_RATIO
    hm = np.zeros((output_h, output_w, cfgs.NUM_CLASS), dtype=np.float32)
    wh = np.zeros((cfgs.MAX_OBJ, 2), dtype=np.float32)
    reg = np.zeros((cfgs.MAX_OBJ, 2), dtype=np.float32)
    reg_mask = np.zeros((cfgs.MAX_OBJ), dtype=np.float32)
    ind = np.zeros((cfgs.MAX_OBJ), dtype=np.float32)
    cls = np.zeros((cfgs.MAX_OBJ), dtype=np.float32)

    for idx, label in enumerate(labels):
        if label[-1] < cfgs.NUM_CLASS:
            bbox = label[:4] / cfgs.DOWN_RATIO
            class_id = label[4]
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            radius = gaussian_radius((math.ceil(h), math.ceil(w)))
            radius = max(0, int(radius))
            ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[:, :, class_id], ct_int, radius)
            wh[idx] = 1. * w, 1. * h
            reg[idx] = ct - ct_int
            reg_mask[idx] = 1
            ind[idx] = ct_int[1] * output_w + ct_int[0]
        cls[idx] = label[-1]

    return image, hm, wh, reg, reg_mask, ind, cls


def get_data(batch_lines, use_aug):
    batch_image = np.zeros((cfgs.BATCH_SIZE, cfgs.INPUT_IMAGE_H, cfgs.INPUT_IMAGE_W, 3), dtype=np.float32)
    # 目标的高斯分布，表示目标的中心点
    batch_hm = np.zeros(
        (cfgs.BATCH_SIZE, cfgs.INPUT_IMAGE_H // cfgs.DOWN_RATIO, cfgs.INPUT_IMAGE_W // cfgs.DOWN_RATIO, cfgs.NUM_CLASS),
        dtype=np.float32)
    # 目标的高度和宽度
    batch_wh = np.zeros((cfgs.BATCH_SIZE, cfgs.MAX_OBJ, 2), dtype=np.float32)
    # 目标中心整数化时的量化误差
    batch_reg = np.zeros((cfgs.BATCH_SIZE, cfgs.MAX_OBJ, 2), dtype=np.float32)
    # 1有目标 0没有目标
    batch_reg_mask = np.zeros((cfgs.BATCH_SIZE, cfgs.MAX_OBJ), dtype=np.float32)
    # 目标关键点在2D heatmap中对应的1D heatmap的索引
    batch_ind = np.zeros((cfgs.BATCH_SIZE, cfgs.MAX_OBJ), dtype=np.float32)
    # 分类标签
    batch_cls = np.zeros((cfgs.BATCH_SIZE, cfgs.MAX_OBJ), dtype=np.float32)
    # batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes= [], [], [], [], [], [], []
    for num, line in enumerate(batch_lines):
        image, hm, wh, reg, reg_mask, ind, cls = process_data(line, use_aug)
        batch_image[num, :, :, :] = image
        batch_hm[num, :, :, :] = hm
        batch_wh[num, :, :] = wh
        batch_reg[num, :, :] = reg
        batch_reg_mask[num, :] = reg_mask
        batch_ind[num, :] = ind
        batch_cls[num, :] = cls

    return batch_image, batch_hm, batch_wh, batch_reg, batch_reg_mask, batch_ind, batch_cls
