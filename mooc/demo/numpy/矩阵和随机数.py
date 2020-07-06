"""
矩阵 matrix
np.mat('1 2 3; 4 5 6')

a = np.array([[1,2,3],[4,5,6]])
m=np.mat(a)

矩阵对象的属性
.ndim 矩阵的维数
.shape 矩阵的形状
.size 矩阵的元素个数
.dtype 元素的数据类型

矩阵相乘
*
转置 .T
求逆 .I

矩阵          VS  二维数组
运算符号简单       能够表示高维数组
A*B              数组更加灵活 速度更快



随机数模块 numpy.random

np.random.rand()     [0,1)
np.random.uniform(low,high,size) [low,high)
numpy.random.randint(low,high,size)
np.random.randn(d0,d1,...dn) 符合标准正态分布
np.random.normal(loc,scale,size) 符合正态分布 均值为0 方差为1

np.random.seed(612)

打乱顺序函数
np.random.shuffle(arr)
"""