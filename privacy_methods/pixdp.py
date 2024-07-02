import cv2
import numpy as np
import math


def pixelation(im_path, is_rgb=False, m=16, epsilon=0.5, b = 16, delta_p = 255):
    '''
    Generate a private image using the Pixelation mechanism

    Keyword arguments:
    im_path -- The path to the image you want to use as input
    is_rgb  -- If using RGB images (default is grayscale)
    m       -- Number of pixels to be protected by the mechanism
    epsilon -- Privacy parameter
    b       -- Algorithm parameter controlling the size of blocks used for pixelation
    delta_p -- Maximum difference between 2 pixel intensities
    '''
    imArray = cv2.imread(im_path)
    imArray = cv2.cvtColor(imArray, cv2.COLOR_BGR2GRAY) if not is_rgb else imArray

    vals = imArray.shape

    if len(vals) ==2:
        h,w = vals
        channels = 1
    elif len(vals) ==3:
        h, w, channels = vals

    if m > h*w:  ## if m is larger than image size
        m = h*w


    h1 = math.ceil(h/b)
    w1 = math.ceil(w/b)


    ## basic blur NP
    blur = cv2.resize(imArray, (w1,h1), interpolation = cv2.INTER_NEAREST)


    ## DP blur

    loc, scale = 0, delta_p*m*channels/(b*b*epsilon)
    blur_dp = blur.copy()


    i=0
    j=0
    while i<h1:
        while j<w1:
            avg = blur_dp[i,j]
            s = np.random.laplace(loc, scale, channels)
            noisy_avg = avg + s
            for elem in np.nditer(noisy_avg, op_flags=['readwrite']):

                if elem > 255:
                    elem[...] = 255
                elif elem < 0:
                    elem[...] = 0
                else:
                    elem[...] = int(elem)

            blur_dp[i,j] = noisy_avg
            j=j+1
        i=i+1
        j=0



    ## median filter as a post-processing step
    median = cv2.medianBlur(blur_dp,3)

    return blur_dp, median, blur