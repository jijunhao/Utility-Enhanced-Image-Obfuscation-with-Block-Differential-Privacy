import numpy as np
import math
from scipy.special import gammaln,gamma
from scipy.stats import entropy
from PIL import Image
from scipy.optimize import fsolve, newton
import random
from decimal import *


def JS_divergence(P, Q):
    _P = P / np.sum(P)
    _Q = Q / np.sum(Q)
    M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, M) + entropy(_Q, M))

def histogram(image, bins=256):
    hist = np.histogram(image, bins=bins, range=(0, 256))[0]
    return hist / np.sum(hist)

def d(x, z, alpha, beta):
    d_e = np.linalg.norm(x - z)**2
    P = histogram(x)
    Q = histogram(z)
    d_js = JS_divergence(P, Q)
    return alpha * d_e + beta * d_js

def K(x, z, epsilon):
    c , n , m = x.shape
    d_val = np.linalg.norm(x - z)**2
    return math.exp(-epsilon * d_val / (c * n * m))

def M(x, z, epsilon, alpha, beta):
    d_val = d(x, z, alpha, beta)
    return math.exp(-epsilon * d_val)

def KM(x, z, epsilon, alpha, beta):
    P = histogram(x)
    Q = histogram(z)
    d_js = JS_divergence(P, Q)
    #print("JS散度:",d_js)
    return math.exp(-epsilon * beta * d_js)


def equation(p, *data):
    n, z = data

    with localcontext() as ctx:
        ctx.prec = 64
        y = Decimal(p[0].astype(float))
        sumation = 0
        for i in range(n):
            sumation += ctx.power(y, i) / math.factorial(i)

        remainder = y.exp() * (1 - Decimal(z)) - sumation

    return float(remainder)


def sampling(n, epsilon, estimate=1):
    z = random.random()
    data = (n, z)

    while True:
        y = fsolve(equation, estimate, args=data, maxfev=5000)
        if y > 0:
            break
        else:
            estimate += 100

    x = y / epsilon

    p_sum = 0
    u = []

    for i in range(n):
        num = random.gauss(0, 1)
        u.append(num)
        p_sum += num ** 2

    for i in range(len(u)):
        u[i] = u[i] / math.sqrt(p_sum)

    return u * x

def sample_from_distribution(x, epsilon):
    c,n,m =x.shape
    z = np.random.normal(x, np.sqrt(c*n*m / (2 * epsilon)), x.shape)
    samples = np.clip(z, 0, 255).astype(np.uint8)
    return samples


def perturb_image(x, epsilon=1):
    c, n, m = x.shape  # 获取输入的形状

    alpha = 1/(c*n*m)  # 根据Theorem 1设置alpha
    beta = (c*n*m - 1)/(c*n*m)  # 根据Theorem 1设置beta
    #t = math.exp(-epsilon * (c * n * m - 1) / (c * n * m))

    while True:
        z = sample_from_distribution(x, epsilon)
        u = np.random.uniform(0, 1)  # 生成一个随机数u
        #fac = t * K(x, z, epsilon) / M(x, z, epsilon, alpha, beta)
        fac = KM(x, z, epsilon, alpha, beta)
        if u <= fac:
            return z


# 示例用法
img_path = "/home/jijunhao/DMDP/datasets/image/image_512_downsampled_from_hq_1024/1.jpg"
img = Image.open(img_path).convert('RGB')
img_array = np.array(img)
img_array = np.transpose(img_array, (2, 0, 1))

perturbed_image = perturb_image(img_array, epsilon=10).astype(np.uint8)

perturbed_image = np.transpose(perturbed_image, (1, 2, 0))
perturbed_img = Image.fromarray(perturbed_image)

# 保存图像
perturbed_img.save("perturbed_1.jpg")
