"""
Matplotlib 绘图库

Figure对象
figure(num,figsize,dpi,facecolor,edgecolor,frameon)
* num 图形编号或名称 取值为数字/字符串
* figsize 绘图对象的宽和高 单位为英寸
* dpi 绘图对象的分辨率 默认每英寸80像素
* facecolor 背景颜色

plt.figure(figsize=(3,2),facecolor='green')
plt.plot()
plt.show()

颜色 缩略字符
blue b black k green g white w red r cyan c yellow y magenta m

figure 对象 划分子图
subplot(行数，列数，子图序号)
plt.subplot(2,2,1)
plt.subplot(2,2,2)
plt.subplot(2,2,3)
plt.subplot(2,2,4)
当行数列数序号均小于10可以省略逗号

设置中文字体
plt.rcParams["font.sans-serif"]="SimHei"

中文字体 英文描述
宋体   SimSun
黑体  SimHei
微软雅黑 MicrosoftYaHei
微软正黑体 MicrosoftJhengHei
楷体 KaiTi
仿宋 FangSong
隶书 LiSu
幼圆 YouYuan

plt.rcdefaults() 恢复标准默认配置

添加标题
添加全局标题
suptitle(标题文字)

suptitle()函数的主要参数
x 标题位置的 x 坐标 0.5
y 标题位置的 y 坐标 0.98
color 标题颜色 黑色
backgroundcolor 标题背景颜色 12
fontsize 标题的字体大小
fontweight 字体粗细 normal
fontstyle 设置字体类型 normal/italic/oblique
horizontalalignment 标题水平对齐方式 center left riht center
verticalalignment 标题的垂直对齐方式 top center top bottom baseline


fontsize: xx-small x-small small medium large x-large xx-large
fontweight； light normal medium semibold bold heavy black


title(标题文字)
title（）函数的主要参数
 loc 标题位置 left,right
rotation 标题文字旋转角度
color fontsize fontweight fontstyle horizontalalignment verticalalignment
fontdict 设置参数字典

tight_layout(ret=[left,bottom,right,top])函数 0011
检查坐标轴标签、刻度标签、和子图标题，自动调整子图，使之填充整个绘图区域，并消除子图之间的重叠。

"""
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Arial Unicode MS']
plt.subplot(221)
plt.title('子标题1')
plt.subplot(222)
plt.title('子标题2')
plt.subplot(223)
myfontdict = {"fontsize": 12, "color": "g", "rotation": 30}
plt.title('子标题3', fontdict=myfontdict)
plt.subplot(224)

plt.suptitle('主标题', fontsize=20)
plt.tight_layout(rect=[0, 0, 1, 0.7])
plt.show()
