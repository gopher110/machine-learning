"""
NumPy Numeric Python
array([列表]/[元组])
a=np.array([0,1,2,3])
type(a) numpy.ndarray

数组的属性
ndim 数组的维度
shape 数组的形状
size 数组元素的总个数
dtype 数组中元素的数据类型
itemsize 数组中每个元素的字节数

数组元素的数据类型
array([列表]/[元组],dtype=np.int64)

创建特殊的数组
np.arange(起始数字，结束数字，步长，dtype) 创建数字序列数组
np.ones(shape,dtype=数据类型) 创建全1数组
np.zeros(shape,dtype=数据类型) 创建全0数组
np.eye(shape) 创建单位矩阵
np.linspace(start,stop,num=50) 创建等差数列
np.logspace(起始指数,结束指数,num=5,base=2) 创建等比数列


asarray() 函数 将列表或元组转化为数组对象
当数据源本身已经是一个 ndarray 对象时
array 仍然会复制出一个副本
asarray() 则直接引用了本来的数组

"""