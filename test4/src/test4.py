"""
实验3：求数据集linear_data.txt的线性回归方程。要求：(计算机专业做)
1) 损失函数采用均方误差（参数theta个数为2）
2) 计算并输出初始theta时的代价值cost1
3) 对theta执行梯度下降（其中，学习率=0.01，迭代次数=1000）
4) 计算并输出梯度下降优化后更新的theta值和代价值cost2
5) 求线性回归模型，并绘制模型拟合直线
6) 绘制迭代次数与cost2的梯度下降曲线
7) 对5）和6）的结果进行分析
"""

import pandas
import numpy

# 图形库
import matplotlib.pyplot
from mpl_toolkits.mplot3d import Axes3D

# 设置该脚本中显示完整数据
numpy.set_printoptions(threshold=numpy.inf)


# 定义损失函数
# 1) 损失函数采用均方误差（参数theta个数为2）
def cost(t_0, t_1, data):
    yi = t_0 + t_1 * data[:, 0]
    j = sum((yi - data[:, 1]) ** 2) / (2 * len(data[:, 1]))
    return j


# 2) 计算并输出初始theta时的代价值cost1

# 读取 data/linear_data.txt 文件
linear_data_path = '../data/linear_data.txt'
linear_data = pandas.read_table(linear_data_path, names=('x', 'y'), sep='[,|\n]')
linear_data = numpy.array(linear_data)

t0 = 0
t1 = 0

# 学习率
a = 0.01

print('初始代价:\n{0}\n'.format(cost(t0, t1, linear_data)))

# 3) 对theta执行梯度下降（其中，学习率=0.01，迭代次数=1000）
aa = []
for i in range(1000):
    temp0 = t0 - a * (1 / len(linear_data[:, 0])) * (sum((t0 + t1 * linear_data[:, 0]) - linear_data[:, 1]))
    t1 = t1 - a * (1 / len(linear_data[:, 0])) * (
        sum(((t0 + t1 * linear_data[:, 0]) - linear_data[:, 1]) * linear_data[:, 0]))
    t0 = temp0
    aa += [[i + 1, cost(t0, t1, linear_data)]]

# 4) 计算并输出梯度下降优化后更新的theta值和代价值cost2
print('第{0}次训练，代价为{1},\nt0:{2},\nt1:{3}\n'.format(i + 1, cost(t0, t1, linear_data), t0, t1))

# 5) 求线性回归模型，并绘制模型拟合直线
aa = numpy.array(aa)
axes = matplotlib.pyplot.subplot()
axes.plot(linear_data[:, 0], linear_data[:, 1], 'o')
axes.plot([0, 30], [t0, t0 + 30 * t1], 'r')
axes.set_title('预测利润 VS 人口数')
matplotlib.pyplot.show()

fig1 = matplotlib.pyplot.figure()

# 6) 绘制迭代次数与cost2的梯度下降曲线
axes = matplotlib.pyplot.subplot()
axes.plot(aa[:, 0], aa[:, 1], 'r')
axes.set_title('误差 VS 预测次数')
matplotlib.pyplot.show()

fig2 = matplotlib.pyplot.figure()

# 7) 对5）和6）的结果进行分析
ax = Axes3D(fig1)
ax = Axes3D(fig2)
x = linear_data
y = aa
x, y = numpy.meshgrid(x, y)
z = numpy.power(x, 2) + numpy.power(y, 2)
ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=matplotlib.pyplot.cm.coolwarm)
matplotlib.pyplot.show()
