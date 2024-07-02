"""
-*- coding: utf-8 -*-

@Author : 季俊豪
@Time : 2024/6/23 下午2:21
@Software: PyCharm 
@File : plotvisual.py
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 11.5})
# x轴 \(\epsilon\) 值
epsilon_values = [0.1, 0.5, 1]

# y轴 PSNR 值
psnr_ours_8 = [31.223, 33.822, 34.770]
psnr_ours_16 = [27.675, 32.132, 32.941]
psnr_pixdp = [14.468, 17.420, 17.556]

# 创建绘图
plt.figure(figsize=(5, 4))

# 绘制数据
plt.plot(epsilon_values, psnr_ours_8, marker='o', label='Ours (block = 8)')
plt.plot(epsilon_values, psnr_ours_16, marker='s', label='Ours (block = 16)')
plt.plot(epsilon_values, psnr_pixdp, marker='^', label='Pix-DP')

plt.xlabel(r'$\epsilon$')
plt.ylabel('PSNR')

# 去除右边框和上边框
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3, frameon=False)
# 设置x轴刻度
plt.xticks([0.1, 0.5, 1])
plt.yticks([15, 20, 25, 30, 35])
plt.grid(linestyle='--', alpha=0.7)
plt.savefig('psnr.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
# 显示图表
plt.show()


import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Times New Roman')

# x轴 \(\epsilon\) 值
epsilon_values = [0.1, 0.5, 1]

# y轴 SSIM 值
ssim_ours_8 = [0.980, 0.988, 0.990]
ssim_ours_16 = [0.968, 0.987, 0.989]
ssim_pixdp = [0.460, 0.709, 0.732]

# 创建绘图
plt.figure(figsize=(5, 4))

# 绘制数据
plt.plot(epsilon_values, ssim_ours_8, marker='o', label='Ours (block = 8)')
plt.plot(epsilon_values, ssim_ours_16, marker='s', label='Ours (block = 16)')
plt.plot(epsilon_values, ssim_pixdp, marker='^', label='Pix-DP')

plt.xlabel(r'$\epsilon$')
plt.ylabel('SSIM')

# 去除右边框和上边框
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(linestyle='--', alpha=0.7)
# 设置图例在图表上方一行
plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=3, frameon=False)

# 设置x轴刻度
plt.xticks([0.1, 0.5, 1])

# 设置y轴刻度
plt.yticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

# 保存图表为PNG文件，并去除白边
plt.savefig('ssim.png', bbox_inches='tight', pad_inches=0.1, dpi=300)

# 显示图表
plt.show()
