import matplotlib.pyplot as plt
import numpy as np

# 创建数据
categories = ['A', 'B', 'C']
value1 = [10, 20, 30]  # 主坐标轴的数据
value2 = [100, 200, 300]  # 副坐标轴的数据1
value3 = [5, 15, 25]  # 副坐标轴的数据2

x = np.arange(len(categories))  # x轴位置

# 创建一个图形和主坐标轴
fig, ax1 = plt.subplots()

# 绘制主坐标轴的柱状图
ax1.bar(x - 0.2, value1, 0.4, label='Value 1', color='skyblue')
ax1.set_xlabel('Category')
ax1.set_ylabel('Primary Axis', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')

# 创建副坐标轴
ax2 = ax1.twinx()  # 共享x轴
ax2.bar(x + 0.2, value2, 0.4, label='Value 2', color='orange', alpha=0.7)
ax2.bar(x + 0.6, value3, 0.4, label='Value 3', color='green', alpha=0.7)
ax2.set_ylabel('Secondary Axis', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 设置 x轴 的刻度
ax1.set_xticks(x)
ax1.set_xticklabels(categories)

# 设置图例
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# 显示图形
plt.title("Bar Plot with Dual Axes")
plt.show()
