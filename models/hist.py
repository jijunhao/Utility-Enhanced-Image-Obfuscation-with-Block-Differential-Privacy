"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2024/6/23 下午6:34
@Software: PyCharm 
@File : hist.py
"""
import cv2
import numpy as np
from scipy.stats import wasserstein_distance
import os


import os
import warnings
warnings.filterwarnings("ignore")

import torch.nn.functional as F
from PIL import Image
import lpips
from torchmetrics import SSIM, PSNR
import torch
from torchvision.transforms import Compose
from torchvision.transforms import ToTensor, Resize
from torchvision.transforms import ToTensor,Normalize, CenterCrop
torch.manual_seed(42)

from tqdm import tqdm
bd = []
wd = []
iou = []


def calculate_histogram_iou(hist1, hist2):
    # 计算直方图交集
    intersection = np.sum(np.minimum(hist1, hist2))
    # 计算直方图并集
    union = np.sum(np.maximum(hist1, hist2))
    # 计算交并比
    iou = intersection / union
    return iou


def calculate_wasserstein_distance(hist1, hist2):
    # 计算 Wasserstein 距离
    return wasserstein_distance(hist1, hist2)


def calculate_bhattacharyya_distance(hist1, hist2):
    # 计算巴氏系数
    bc = np.sum(np.sqrt(hist1 * hist2))
    # 计算巴氏距离
    bd = -np.log(bc)
    return bd


def compute_histograms(img1, img2, bins=256):
    # 计算两个图像的直方图并归一化
    hist1 = cv2.calcHist([img1], [0], None, [bins], [0, 256])
    hist2 = cv2.calcHist([img2], [0], None, [bins], [0, 256])

    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    return hist1.flatten(), hist2.flatten()


ori_path = "/home/jijunhao/DMDP/datasets/image2"

pro_path = "/home/jijunhao/DPINDEX/output/pixdp_epsilon_01"

files = os.listdir(pro_path)

for file_name in tqdm(files[:1000]):
    if file_name.endswith(".png") or file_name.endswith(".jpg"):

        # 读取两张示例图像
        img1 = cv2.imread(os.path.join(ori_path, file_name))
        img2 = cv2.imread(os.path.join(pro_path, file_name.split('.')[0] + '.jpg'))

        # 计算直方图
        hist1, hist2 = compute_histograms(img1, img2)

        # 计算巴氏距离
        bhattacharyya_distance = calculate_bhattacharyya_distance(hist1, hist2)
        bd.append(bhattacharyya_distance)

        # 计算直方图交并比（IoU）
        histogram_iou = calculate_histogram_iou(hist1, hist2)
        iou.append(histogram_iou)


bd = sum(bd)/len(bd)
iou = sum(iou)/len(iou)


print(f"巴氏距离: {bd:.3f}")
print(f"IoU: {iou:.3f}")

