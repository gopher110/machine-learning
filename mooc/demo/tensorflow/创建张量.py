"""
TensorFlow中的 Tensor 表示张量，其实就是多维数组
Python list
NumPy中的数组对象 ndarray
它们也都可以作为数据的载体

Python 列表（list）:
    元素可以使用不同的数据类型,可以嵌套
    在内存中不连续存放，是一个动态的指针数组
    读写效率低，占用内存空间大
    不适合做数值计算
NumPy数组（ndarray）:
    元素的数据类型相同
    每个元素在内存中占用的空间相同，存储在一个连续的内存区域中
    存储空间小，读取和写入速度快
    在 CPU 中运算，不能够主动检测、利用 GPU 进行计算
TensorFlow 张量（Tensor）：
    可以高速运行于 GPU 和 TPU 之上
    支持 CPU、嵌入式、单机多卡、多机多卡等多种计算环境
    高速的实现神经网络和深度学习中复杂算法

TensorFlow 的基本运算、参数命名、运算规则、API 的设计等与 NumPy 非常相近

创建Tensor对象
张量由 Tensor 类实现，每个张量都是一个 Tensor 对象
tf.constant()函数 创建张量
tf.constant(value,dtype,shape)
value: 数字，python 列表，NumPy 数组 布尔型
dtype 元素的数据类型
shape 张量的形状
tensor 张量是对 numpy array 的封装
tf.constant([[1, 2], [3, 4]])
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32)

 张量的.numpy()方法
numpy 创建浮点数数组时，默认的浮点型是64位浮点数
当使用 NumPy 数组创建张量的时候，TensorFlow 会接受数组元素的数据类型，使用64位浮点数保存数据

tf.cast(x,dtype)改变张量中元素的数据类型

tf.convert_to_tensor()函数
参数 数组、列表、数字、布尔值、字符串
na = np.arrange(12).reshape(3,4)
ta=tf.convert_to_tensor(na)

is_tensor()函数 判断是否为张量 tf.is_tensor(ta)
py isinstance()函数 isinstance(ta,tf.Tensor) na ndarray
"""

"""
创建全0和全1张量
tf.zeros(shape,dtype=tf.float32)
tf.ones(shape,dtype=tf.float32)

tf.fill()函数 创建元素值都相同的张量
tf.fill(dims,value) tf.fill([2,3],9)
也可以通过 constant创建
tf.constant(9,shape=[2,3])

创建随机数张量 正态分布
tf.random.normal(shape,mean,stddev,dtype)
tf.random.normal([2,2])创建一个2x2的张量，其元素服从标准正态分布
tf.random.normal([3,3,3],mean=0.0,stddev=2.0)

截断正态分布
tf.random.truncated_normal(shape,mean,stddev,dtype)
返回一个截断的正态分布
截断的标准是2倍的标准差

创建均匀分布张量 tf.random.uniform()函数
tf.random.uniform(shape,minval,maxval,dtype) [minval,maxval)

随机打乱 tf.random.shuffle()函数

创建序列 tf.range(start,limit,delta=1,dtype)函数 [start,limit)

Tensor 对象的属性 ndim shape dtype
tf.shape(a) tf.size(a) tf.rank(a)

张量和 NumPy 数组
在 TensorFlow 中 所有的运算都是在张量之间进行的
NumPy 数组仅仅是作为输入和输出来使用
张量可以运行于 CPU 也可以运行于 GPU和 TPU
而 NumPy数组只能够在 CPU 中运行

"""
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
a = tf.constant([[1, 2], [3, 4]])
print(type(a))
a.numpy()
print(tf.constant([[1, 2], [3, 4]]))
print(type(a))  # <class 'tensorflow.python.framework.ops.EagerTensor'>
print(tf.constant(1.0, dtype=tf.float64))
print(tf.constant(np.array([1.0, 2.0]), dtype=tf.float32))