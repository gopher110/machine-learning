"""
起始位置：结束位置：步长

数据提取
gather()函数
用一个索引列表 将给定张量中 对应索引值的元素提取出来
gather(params,indices,axis)

a=tf.range(5)
tf.gather(a,indices=[0,2,3])


同时采样多个点  gather_nd()函数
tf.gather_nd(n,[[0,0],[1,1],[2,3]])

"""