"""
get_file()函数 -- 下载数据集
tf.keras.utils.get_file(fname,origin,cache_dir)
参数
fname：下载后的文件名
origin：文件的 URL 地址
cache_dir：下载后文件的存储位置

Pandas库
Panel Data & Data Analysis
用于数据统计和分析
可以高效、方便地操作大型数据集

pd.read_csv()

DataFrame 常用属性
ndim 数据表的维数
shape 数据表的形状
size 数据表元素的总个数


scatter(x,y,c,cmap)
色彩映射
将参数 c 指定为一个列表或数组，所绘制的图形的颜色,可以随这个列表
或数组中元素的值而变换，变换所对应的颜色由参数 cmap 中的颜色所提供。


"""
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)

COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
df_iris = pd.read_csv(train_path, names=COLUMN_NAMES, header=0)
print(df_iris.describe())

iris = df_iris.values
print(type(iris))
iris = np.array(df_iris)
print(type(iris))


fig = plt.figure('Iris Data', figsize=(15, 15))
fig.suptitle("Anderson's Iris Data Set\n(Bule->Setosa | Red->Versicolor | Green->Virginica)")
for i in range(4):
    for j in range(4):
        plt.subplot(4, 4, 4*i+j+1)
        if i == j:
            plt.text(0.3, 0.4, COLUMN_NAMES[0], fontsize=15)
        else:
            plt.scatter(iris[:, j], iris[:, i], c=iris[:, 4], cmap='brg')
        if i == 0:
            plt.title(COLUMN_NAMES[j])
        if j == 0:
            plt.ylabel(COLUMN_NAMES[i])



plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()