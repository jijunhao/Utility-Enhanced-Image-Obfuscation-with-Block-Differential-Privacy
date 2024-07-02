import numpy as np
from PIL import Image, ImageDraw
from tqdm import trange
from collections import defaultdict
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


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

if __name__ == '__main__':
    block = 16
    epsilon1 = 0.1
    epsilon2 = 0.1


    image_dataset_folder = '/home/jijunhao/DPINDEX/my'

    output_folder = './output/my/'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 列出文件夹中的所有文件
    image_dataset_names = os.listdir(image_dataset_folder)

    for image_data in tqdm(image_dataset_names[:10]):
        # 构建图像文件的完整路径
        file_path = os.path.join(image_dataset_folder, image_data)

        img = Image.open(file_path).convert('RGB')

        width, height = img.size
        new_width = (width // block) * block
        new_height = (height // block) * block

        img = img.resize((new_width, new_height))
        width, height = img.size


        img_array = np.array(img)

        for x in range(0, width, block):
            for y in range(0, height, block):

                block_data = img_array[y:y + block, x:x + block, :]

                # 统计0-255的分别个数并添加噪声
                for i in range(3):  # 对于每个颜色通道
                    count = np.bincount(block_data[:, :, i].flatten(), minlength=256)
                    non_zero_indices = np.nonzero(count)[0]
                    dpvalues = np.array([laplace(count[val], 1, epsilon1) for val in non_zero_indices])
                    dpvalues = np.clip(dpvalues, 1e-10, None)  # Ensure no zero probabilities
                    probabilities = dpvalues / np.sum(dpvalues)

                    # count = np.histogram(block_data[:, :, i], bins=256, range=(0, 256))[0]
                    # dpvalues = count + np.random.laplace(0, 1/epsilon1, size=count.shape)
                    # dpvalues = np.clip(dpvalues, 1e-10, None)  # Ensure no zero probabilities
                    # probabilities = dpvalues / np.sum(dpvalues)


                    sampled_values_list = []
                    for _ in range(10):
                        #sampled_values = np.random.choice(range(256), size=block_data[:, :, i].shape, p=probabilities)
                        sampled_values = np.random.choice(non_zero_indices, size=block_data[:, :, i].shape, p=probabilities)
                        sampled_values_list.append(sampled_values)
                    sampled_values_array = np.array(sampled_values_list)

                    select_best = problib(block_data[:, :, i], sampled_values_array, 255, epsilon=epsilon2)
                    block_data[:, :, i] = select_best


                img_array[y:y + block, x:x + block, :] = block_data


        # 创建替换后的图像
        new_img = Image.fromarray(np.uint8(img_array))
        new_img.save(os.path.join(output_folder, image_data))
