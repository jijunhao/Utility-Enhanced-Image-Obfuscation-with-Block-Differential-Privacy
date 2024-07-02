"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2024/6/23 下午2:55
@Software: PyCharm 
@File : plotpixel.py
"""

from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np

# 设置路径
ori_path = '/home/jijunhao/DPINDEX/output/helen/our_block_16_epsilon_01'
file_name = '1002681492_1.jpg'
thumbnail_path = 'thumbnail_' + file_name

# 打开图像并转换为RGB模式
img = Image.open(os.path.join(ori_path, file_name)).convert('RGB')

# 定义块大小
block_size = 16

# 创建块像素化图像
pixelated_img = img.copy()
pixels = pixelated_img.load()

for i in range(0, img.size[0], block_size):
    for j in range(0, img.size[1], block_size):
        # 计算每个块的平均值
        r, g, b = [], [], []
        for x in range(i, min(i + block_size, img.size[0])):
            for y in range(j, min(j + block_size, img.size[1])):
                r.append(pixels[x, y][0])
                g.append(pixels[x, y][1])
                b.append(pixels[x, y][2])
        r_avg = int(np.mean(r))
        g_avg = int(np.mean(g))
        b_avg = int(np.mean(b))
        # 将块的像素值设置为平均值
        for x in range(i, min(i + block_size, img.size[0])):
            for y in range(j, min(j + block_size, img.size[1])):
                pixels[x, y] = (r_avg, g_avg, b_avg)


img = img.resize((img.size[1], img.size[1]))
pixelated_img = pixelated_img.resize((img.size[1], img.size[1]))

# 展示原图像
plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.axis('off')
plt.show()
img.save(file_name, bbox_inches='tight', pad_inches=0)


# 展示原图像
plt.figure(figsize=(6, 6))
plt.imshow(pixelated_img)
plt.axis('off')
plt.show()

# 保存块像素化图像
pixelated_img.save(thumbnail_path, bbox_inches='tight', pad_inches=0)