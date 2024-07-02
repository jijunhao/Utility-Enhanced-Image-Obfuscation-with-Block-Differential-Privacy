import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 11.5})
# x轴 \(\epsilon\) 值
epsilon_values = [0.1, 0.5, 1]

# y轴 巴氏距离值
bhattacharyya_ours_8 = [0.008, 0.007, 0.007]
bhattacharyya_ours_16 = [0.018, 0.015, 0.014]
bhattacharyya_ours_32 = [0.039, 0.032, 0.031]
bhattacharyya_pixdp = [0.153, 0.134, 0.126]

# 创建绘图
plt.figure(figsize=(5, 5))

# 绘制数据
plt.plot(epsilon_values, bhattacharyya_ours_8, marker='o', label='Ours (block = 8)')
plt.plot(epsilon_values, bhattacharyya_ours_16, marker='s', label='Ours (block = 16)')
plt.plot(epsilon_values, bhattacharyya_ours_32, marker='^', label='Ours (block = 32)')
plt.plot(epsilon_values, bhattacharyya_pixdp, marker='x', label='Pix-DP')

plt.xlabel(r'$\epsilon$')
plt.ylabel('Bhattacharyya Distance')

# 去除右边框和上边框
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2, frameon=False)
plt.grid(linestyle='--', alpha=0.7)
# 设置x轴刻度
plt.xticks([0.1, 0.5, 1])
plt.yticks(np.linspace(0, 0.16, 9))
plt.savefig('bhattacharyya_distance.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
# 显示图表
plt.show()

# 绘制 IoU 图
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', family='Times New Roman')
plt.rcParams.update({'font.size': 11.5})

# y轴 IoU 值
iou_ours_8 = [0.847, 0.853, 0.855]
iou_ours_16 = [0.767, 0.790, 0.794]
iou_ours_32 = [0.667, 0.700, 0.703]
iou_pixdp = [0.458, 0.493, 0.498]

# 创建绘图
plt.figure(figsize=(5, 5))

# 绘制数据
plt.plot(epsilon_values, iou_ours_8, marker='o', label='Ours (block = 8)')
plt.plot(epsilon_values, iou_ours_16, marker='s', label='Ours (block = 16)')
plt.plot(epsilon_values, iou_ours_32, marker='^', label='Ours (block = 32)')
plt.plot(epsilon_values, iou_pixdp, marker='x', label='Pix-DP')

plt.xlabel(r'$\epsilon$')
plt.ylabel('IoU')

# 去除右边框和上边框
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(linestyle='--', alpha=0.7)
plt.legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=2, frameon=False)
# 设置x轴刻度
plt.xticks([0.1, 0.5, 1])
plt.yticks(np.linspace(0.4, 0.9, 6))
plt.savefig('iou.png', bbox_inches='tight', pad_inches=0.1, dpi=300)
# 显示图表
plt.show()
