"""
当改变形状时，应该考虑到数组中元素的个数，确保改变前后 元素总个数相等
np.reshape(shape) 不改变当前数组 按照 shape 创建新的数组
np.resize(shape) 改变当前数组，按照 shape 创建数组

b=np.arrange(12)
array([0,1,2,3,4,5,6,7,8,9,10,11])

b.reshape(3,4)
array([[0,1,2,3],[4,5,6,7],[8,9,10,11]])

b array([0,1,2,3,4,5,6,7,8,9,10,11])

b.resize(3,4)
b array([0,1,2,3],[4,5,6,7],[8,9,10,11])

np.arange(12).reshape(3,4)
array([[0,1,2,3],
      [4,5,6,7],
      [8,9,10,11]])

b.reshape(-1,1) 12*1
b.reshape(-1)一维数组

b**2

矩阵运算---矩阵乘法
乘号运算符：矩阵中对应元素分别相乘

矩阵相乘 A X B np.matmul(A,B) np.dot(A,B)
矩阵转置 np.transpose()
矩阵求逆 np.linalg.inv()

"""

"""
数组间的运算
函数  功能描述
numpy.sum() 计算所有元素的和
numpy.prod 计算所有元素的乘积
numpy.diff 计算数组的相邻元素之间的差
np.sqrt() 计算各元素的平方根
np.exp()计算各元素的指数值
np.abs() 取各元素的绝对值

np.logspace(1,4,4,base=2)
[2 4 8 16]
np.sqrt(d)
[1.414 2 2.8284 4]



一维数组
shape(4,)
a=np.arange(4)
array([0 1 2 3])
np.sum(a)
6

二维数组
b=np.arrange(12).reshape(3,4)
b array([[0,1,2,3],
      [4,5,6,7],
      [8,9,10,11]])
      
np.sum(b,axis=0)     
array([12,15,18,21]) 

np.sum(b,axis=1)     
array([6,22,38]) 

三维数组
t=np.arange(24).reshape(2,3,4) 0,1,2
t array [[[ 0  1  2  3]
  [ 4  5  6  7]
  [ 8  9 10 11]],

 [[12 13 14 15]
  [16 17 18 19]
  [20 21 22 23]]]

np.sum(t,axis=0)
[[12 14 16 18]
 [20 22 24 26]
 [28 30 32 34]]
 
np.sum(t,axis=1)
[[12 15 18 21]
 [48 51 54 57]]

np.sum(t,axis=2)
[[ 6 22 38]
 [54 70 86]]
 
 
数组的堆叠运算
np.stack((数组1 ，数组2，...),axis)
1 2 3
4 5 6
axis=0 shape(2,3)
axis=1 shape(3,2)
"""