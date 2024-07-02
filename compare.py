import os
import cv2
from privacy_methods import pixdp,CODER
from tqdm import tqdm
from PIL import Image
import numpy as np

#image_dataset_folder = '/home/jijunhao/DMDP/datasets/image2'
image_dataset_folder = '/home/jijunhao/DPINDEX/dataset/img'

output_folder = './output/helen/pixdp_epsilon_01/'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 列出文件夹中的所有文件
image_dataset_names = os.listdir(image_dataset_folder)

for image_data in tqdm(image_dataset_names):
    # 构建图像文件的完整路径
    file_path = os.path.join(image_dataset_folder, image_data)


    blur_dp, _, _ = pixdp.pixelation(file_path, m=1, epsilon=0.1, b=8, delta_p=255)

    output_blur_dp_path = os.path.join(output_folder, image_data)

    cv2.imwrite(output_blur_dp_path, blur_dp)

# for image_data in tqdm(image_dataset_names):
#     # 构建图像文件的完整路径
#     file_path = os.path.join(image_dataset_folder, image_data)
#
#     img = Image.open(file_path).convert('RGB')
#     img_array = np.array(img)
#     img_array = np.transpose(img_array, (2, 0, 1))
#
#     perturbed_image = CODER.perturb_image(img_array, epsilon=0.5).astype(np.uint8)
#
#     perturbed_image = np.transpose(perturbed_image, (1, 2, 0))
#     perturbed_img = Image.fromarray(perturbed_image)
#
#     # 保存图像
#     perturbed_img.save(os.path.join(output_folder, image_data))
