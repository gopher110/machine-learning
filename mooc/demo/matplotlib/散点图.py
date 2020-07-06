"""
散点图（Scatter）
原始数据分布的规律 数据变化的趋势

数据分组
Scatter()函数
Scatter(x,y,scale,color,marker,label)
参数    说明
x       数据点 x 坐标
y       数据点 y 坐标
scale   数据点的大小 36
color 数据点的颜色
marker 数据点的样式 o 圆点
label 图例文字


text() 函数
text(x,y,s,fontsize,color)
参数     说明
x 文字的 x 坐标
y 文字的 y 坐标
s 显示的文字
fontsize 文字的大小 12
color 文字的颜色 黑色

xlabel(x,y,s,fontsize,color) 设置 x 轴标签
ylabel(x,y,s,fontsize,color)设置 y 轴标签
xlim(xmin,xmax) 设置 x 轴坐标的范围
ylim(ymin,ymax)设置 y 轴坐标的范围
tick_params(labelsize) 设置刻度文字的字号


增加图例
scatter(x,y,scale,color,marker,label)
legend(loc,fontsize)
loc 参数的取值
0 best
1 upper right
2 upper left
3 lower left
4 lower right
5 right
6 center left
7 center right
8 lower center
9 upper center
10 center

"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

n = 1024
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)

x2 = np.random.uniform(-4, 4, (1, n))
y2 = np.random.uniform(-4, 4, (1, n))

plt.scatter(x, y, color='blue', marker='*', label='正态分布')
plt.scatter(x2, y2, color='yellow', marker='o', label='均匀分布')
plt.text(2.5, 2.5, '均值：0\n 标准差：1')

plt.xlim(-4, 4)
plt.ylim(-4, 4)

plt.xlabel('横坐标 x', fontsize=14)
plt.ylabel('纵坐标 y', fontsize=14)
plt.legend()
plt.title("标准正态分布", fontsize=20)
plt.show()