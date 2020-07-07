"""
基本数学运算
tf.add(x,y)
tf.subtract(x,y)
tf.multiply(x,y)
tf.divide(x,y)
tf.math.mod()

幂指对数
tf.pow(x,y)
tf.square(x)
tf.sqrt(x)
tf.exp(x)
tf.math.log(x) 计算自然对数 底数为 e

log a b = log c b /log c a
log 2 256 = log e 256 / log e 2

其他运算
tf.sign(x) 返回 x 的符号
tf.abs(x) 对 x 逐元素求绝对值
tf.negative(x) 相反数 y= -x
tf.reciprocal(x) 取 x 的倒数
tf.logical_not(x) 逻辑非
tf.ceil(x)  向上取整
tf.floor(x) 向下取整
tf.rint(x) 取最接近的整数
tf.round(x) 舍入最接近的整数
tf.maximum(x,y) 返回两个 tensor 中的最大值
tf.minmum(x,y) 返回两个 tensor 中的最小值

三角函数 反三角函数
tf.cos()
tf.sin()
tf.tan()
tf.acos(x)
tf.asin(x)
tf.atan(x)

广播机制
1,2,3, 2,2,3 最后维度一致

当张量和 NumPy 数组共同参与运算时：
执行 TensorFlow 操作 那么 TensorFlow 将自动把 NumPy 数组转换成张量
执行 Numpy 操作 那么NumPy将自动的把张量转化为 NumPy 数组

使用运算符操作
只要操作数中有一个 tensor 对象，就会把所有的操作数都转化为张量，然后再进行运算。
"""

"""
向量乘法
tf.matmul(a,b)
a@b 

多维向量乘法
2，3，5 @ 5，4 =  2，3，4

数据统计:求张量在某个维度上或者全局的统计值
tf.reduce_sum(input_tensor,axis)求和
tf.reduce_mean(input_tensor,axis)求平均值
tf.reduce_max(tensor,axis)求最大值
tf.reduce_min(tensor,axis)求最小值

求最值的索引
tf.argmax() tf.argmin() 默认 axis=0

1 5 3
2 4 6
tf.argmax(a,axis) [1,0,1]





"""