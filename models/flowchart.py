"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2024/6/19 下午3:21
@Software: PyCharm 
@File : flowchart.py
"""

import numpy as np
from PIL import Image, ImageDraw
from tqdm import trange
from collections import defaultdict
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', family='Times New Roman')

def problib(data, r, sensitivity, epsilon=1.0):

    # 计算R中每个回复的分数
    scores = 255 - np.abs(np.mean(r, axis=(1,2)) - np.mean(data))

    # 根据分数计算每个回复的输出概率
    probabilities = np.exp(epsilon * scores / (2 * sensitivity))

    # 对概率进行归一化处理，使概率和等于1
    probabilities = probabilities / np.sum(probabilities)

    # 根据概率分布选择回复结果
    return r[np.random.choice(np.arange(len(r)), 1, p=probabilities)[0]]

def laplace(x,sensitivity,epsilon=1.0):
    return x + int(np.random.laplace(loc=0, scale=np.array(sensitivity / epsilon), size=x.shape))

# Function to add grid borders to an image
# Function to add grid borders to an image without occupying pixels
def add_grid_borders(image, grid_size=32, border_color=(255, 255, 0), border_width=1):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for x in range(0, width, grid_size):
        draw.line([(x, 0), (x, height)], fill=border_color, width=border_width)
    for y in range(0, height, grid_size):
        draw.line([(0, y), (width, y)], fill=border_color, width=border_width)
    return image

if __name__ == '__main__':
    block = 32
    epsilon1 = 1
    epsilon2 = 1

    img = Image.open("/home/jijunhao/DPINDEX/protected_block_32_epsilon1_1_epsilon2_1.png").convert('RGB')

    # 获取图像尺寸
    width, height = img.size

    # 计算调整后的尺寸
    new_width = (width // block) * block
    new_height = (height // block) * block

    # 调整图像尺寸
    img = img.resize((new_width, new_height))
    width, height = img.size


    # Split the image into red, green, and blue channels
    r, g, b = img.split()

    # Create images with only the respective channel's color
    r_img = Image.merge("RGB", (r, Image.new("L", r.size), Image.new("L", r.size)))
    g_img = Image.merge("RGB", (Image.new("L", g.size), g, Image.new("L", g.size)))
    b_img = Image.merge("RGB", (Image.new("L", b.size), Image.new("L", b.size), b))

    # Save the colored images
    r_img.save("red_channel.jpg")
    g_img.save("green_channel.jpg")
    b_img.save("blue_channel.jpg")


    # Add grid borders to the images
    r_img_with_borders = add_grid_borders(r_img)
    g_img_with_borders = add_grid_borders(g_img)
    b_img_with_borders = add_grid_borders(b_img)

    # Save the images with borders
    r_img_with_borders.save("red_channel_with_borders.jpg")
    g_img_with_borders.save("green_channel_with_borders.jpg")
    b_img_with_borders.save("blue_channel_with_borders.jpg")

    # 分离通道
    r, g, b = img.split()

    # 提取位于第(11, 4)位置的32x32块
    block_x = 11 * block
    block_y = 4 * block
    green_block = g.crop((block_x, block_y, block_x + block, block_y + block))

    # 将图像转换为NumPy数组以便分析像素分布
    green_block_array = np.array(green_block)

    # 将绿色通道块与其他通道合并成RGB图像
    green_rgb_block = Image.merge("RGB",
                                  (Image.new("L", green_block.size), green_block, Image.new("L", green_block.size)))

    # 绘制提取的32x32绿色通道块
    plt.figure(figsize=(6, 6))
    plt.imshow(green_rgb_block, interpolation='nearest')
    plt.axis('off')  # 关闭坐标轴
    plt.savefig("green_block.png", bbox_inches='tight', pad_inches=0)
    plt.show()
    # 计算像素分布
    unique, counts = np.unique(green_block_array, return_counts=True)
    pixel_distribution = dict(zip(unique, counts))

    # 对像素分布添加拉普拉斯噪音
    pixel_distribution_noisy = {}
    for pixel_value, count in pixel_distribution.items():
        noise = np.random.laplace(loc=0.0, scale=1.0)  # 调整噪音规模
        noisy_count = count + noise
        pixel_distribution_noisy[pixel_value] = max(0, noisy_count)  # 确保计数非负

    # 归一化噪音分布
    total_noisy_count = sum(pixel_distribution_noisy.values())
    normalized_noisy_distribution = {k: v / total_noisy_count for k, v in pixel_distribution_noisy.items()}

    # 生成新的采样像素值并保存10张图像
    output_folder = "/home/jijunhao/DPINDEX/output"
    os.makedirs(output_folder, exist_ok=True)

    # 计算原始块的均值
    original_mean = np.mean(green_block_array)
    print(f'Original Block Mean: {original_mean}')

    for i in range(10):
        new_green_block_array = np.random.choice(list(normalized_noisy_distribution.keys()),
                                                 size=(block, block),
                                                 p=list(normalized_noisy_distribution.values()))

        # 计算当前采样块的均值
        current_mean = np.mean(new_green_block_array)
        print(f'Sampled Block {i + 1} Mean: {current_mean}')

        # 将新的绿色通道块与其他通道合并成RGB图像
        new_green_block = Image.fromarray(new_green_block_array.astype(np.uint8))
        new_green_rgb_block = Image.merge("RGB", (
        Image.new("L", green_block.size), new_green_block, Image.new("L", green_block.size)))

        # 保存图像到文件夹
        output_path = os.path.join(output_folder, f"noisy_green_block_{i + 1}.png")
        new_green_rgb_block.save(output_path)

        # 绘制并保存图像以确保正确性
        plt.figure(figsize=(6, 6))
        plt.imshow(new_green_rgb_block, interpolation='nearest')
        plt.axis('off')  # 关闭坐标轴
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()

    # 设置Seaborn调色板
    palette = sns.color_palette("muted")
    # 绘制加噪前的像素分布图并保存
    plt.figure(figsize=(7, 5))
    plt.bar(pixel_distribution.keys(), pixel_distribution.values(), width=1.0, color=palette[0], edgecolor='black')
    plt.xlim(50, 150)
    plt.ylim(0, 25)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # 移除右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig("original_pixel_distribution.png", bbox_inches='tight',dpi=500)
    plt.close()

    # 绘制加噪后的像素分布图并保存
    plt.figure(figsize=(7, 5))
    for value in range(50, 151):
        count_original = pixel_distribution.get(value, 0)
        count_noisy = pixel_distribution_noisy.get(value, 0)

        # 绘制加噪前的柱状图
        plt.bar(value, count_original, width=1.0, color=palette[0], edgecolor='black')

        # 绘制加噪后的变化部分
        if count_noisy != count_original:
            plt.bar(value, min(count_original, count_noisy), width=1.0, color=palette[0], edgecolor='black')
            plt.bar(value, abs(count_noisy - count_original), bottom=min(count_original, count_noisy), width=1.0,
                    color=palette[1], edgecolor='black')

    plt.xlim(50, 150)
    plt.ylim(0, 25)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    # 移除右边框和上边框
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig("noisy_pixel_distribution.png", bbox_inches='tight',dpi=500)
    plt.close()