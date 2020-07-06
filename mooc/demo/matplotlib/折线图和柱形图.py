"""
折线图（Line Chart）
在散点图的基础上，将相邻的点用线段相连接
* 描述变量变化的趋势

plot() 函数
plot(x,y,color,marker,label,linewidth,markersize)
参数       说明
x       数据点的 x 坐标
y       数据点的 y 坐标
color   数据点的颜色
marker  数据点的样式 o 圆点
label 图例文字
linewidth 折线的宽度
markersize 数据点的大小

"""

"""
柱状图（Bar Chart）
由一系列高度不等的柱形条纹表示数据分布的情况。
bar(left,height,width,facecolor,edgecolor,label)



"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial Unicode MS']
n = 24
y1 = np.random.randint(27, 37, n)
y2 = np.random.randint(40, 60, n)

plt.subplot(121)

plt.plot(y1, label='温度')
plt.plot(y2, label='湿度')

plt.xlim(0, 23)
plt.ylim(20, 70)
plt.xlabel('小时', fontsize=12)
plt.ylabel('测量值', fontsize=12)

plt.title('24小时温度湿度统计', fontsize=16)
plt.legend()

plt.subplot(122)
# 柱状图
y1 = [32, 25, 16, 30, 24, 45, 40, 33, 28, 17, 24, 20]
y2 = [-23, -35, -26, -35, -45, -43, -35, -32, -23, -17, -22, -28]
plt.bar(range(len(y1)), y1, width=0.8, facecolor='green', edgecolor='m', label='统计量1')
plt.bar(range(len(y2)), y2, width=0.8, facecolor='red', edgecolor='m', label='统计量2')


plt.show()
