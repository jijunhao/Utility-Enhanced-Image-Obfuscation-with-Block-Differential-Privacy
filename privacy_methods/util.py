# import the necessary packages
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
#import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import lpips

# def mse(imageA, imageB):
#     error = np.mean((imageA - imageB)**2)
#     return error
#
# def compare_images(imageA, imageB):
# 	# compute the mean squared error and structural similarity
# 	# index for the images
# 	m = mse(imageA, imageB)
# 	s = ssim(imageA, imageB)


def calc_psnr(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img1 = img1.resize(img1.size)

    img1 = np.array(img1)
    img2 = np.array(img2)

    psnr_score= psnr(img1, img2, data_range=255)

    return psnr_score



def calc_ssim(img1_path, img2_path):

    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')
    img1, img2 = np.array(img1), np.array(img2)
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score


def calc_mse(img1_path, img2_path):
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    # img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    img1, img2=img1.astype(int), img2.astype(int)
    img1, img2 = img1/256, img2/256

    # MAE = np.mean(abs(img2-img1))
    # RMSE = np.sqrt(np.mean((img2 - img1)**2))
    MSE = np.mean((img2 - img1)**2)
    return MSE


import lpips


class util_of_lpips():
    def __init__(self, use_gpu=False):
        '''
        Parameters
        ----------
        net: str
            抽取特征的网络，['alex', 'vgg']
        use_gpu: bool
        Returns
        -------
        References
        -------
        https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips_2imgs.py

        '''
        ## Initializing the model
        self.loss_fn = lpips.LPIPS(net='vgg')
        self.use_gpu = use_gpu
        if use_gpu:
            self.loss_fn.cuda()

    def calc_lpips(self, img1_path, img2_path):

        # Load images
        img0 = lpips.im2tensor(lpips.load_image(img1_path))  # RGB image from [-1,1]
        img1 = lpips.im2tensor(lpips.load_image(img2_path))

        if self.use_gpu:
            img0 = img0.cuda()
            img1 = img1.cuda()
        lpips = self.loss_fn.forward(img0, img1)
        return lpips

net=util_of_lpips()