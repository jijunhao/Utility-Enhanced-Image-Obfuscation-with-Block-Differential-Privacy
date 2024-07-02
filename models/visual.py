"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2024/6/22 下午11:26
@Software: PyCharm 
@File : visual.py
"""

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


maelist = []
rmselist = []
psnrlist = []
ssimlist = []
lpipslist = []

loss_fn_alex = lpips.LPIPS(net='alex').cuda()  # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization


ori_path = "/home/jijunhao/DMDP/datasets/image2"

pro_path = "/home/jijunhao/DPINDEX/output/pixdp_epsilon_01"
files = os.listdir(pro_path)

for file_name in tqdm(files[:1000]):
    if file_name.endswith(".png") or file_name.endswith(".jpg"):
        img = Image.open(os.path.join(ori_path, file_name)).convert('RGB')
        # 获取原图像的尺寸
        original_size = img.size


        thumbnail_size = (original_size[0] // 8, original_size[1] // 8)

        img.thumbnail(thumbnail_size)
        img_tensor = Compose([
                ToTensor()])(img).unsqueeze(0).cuda()

        pro_img = Image.open(os.path.join(pro_path, file_name.split('.')[0] + '.jpg')).convert('RGB')

        pro_img.thumbnail(thumbnail_size)
        pro_img_tensor = Compose([ToTensor()])(pro_img).unsqueeze(0).cuda()

        if img_tensor.shape==pro_img_tensor.shape:
            #mae = F.l1_loss(img_tensor,pro_img_tensor)
            #rmse = torch.sqrt(F.mse_loss(img_tensor,pro_img_tensor))
            psnr = PSNR().cuda()
            ssim = SSIM().cuda()

            #maelist.append(mae)
            #rmselist.append(rmse)
            psnrlist.append(psnr(img_tensor,pro_img_tensor))
            ssimlist.append(ssim(img_tensor,pro_img_tensor))
            #lpipslist.append(loss_fn_alex(img_tensor,pro_img_tensor).item())


#mae = sum(maelist)/len(maelist)
#rmse = sum(rmselist)/len(rmselist)
psnr = sum(psnrlist)/len(psnrlist)
ssim = sum(ssimlist)/len(ssimlist)
#lpips = sum(lpipslist)/len(lpipslist)

# 打印结果
#print(f'MAE: {mae:.3f}')
#print(f"RMSE: {rmse:.3f}")
print(f"PSNR: {psnr:.3f}")
print(f"SSIM: {ssim:.3f}")
#print(f"LPIPS: {lpips:.3f}")