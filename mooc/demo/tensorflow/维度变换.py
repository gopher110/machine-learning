"""
改变张量的形状
tf.reshape(tensor,shape)
a=tf.range(24)
tf.reshape(a,shape=[2,3,4])
或者
tf.constant(np.arrange(24).reshape(2,3,4))

shape 参数=-1 自动推导出长度

多维张量的轴 ：张量的维度
axis 0 1 2
-3 -2 -1

增加和删除维度
增加维度
tf.expand_dims(input,axis)
增加的这个维度上 长度为1
tf.constant（[1,2]） (2,)
tf.expand_dims(t,1) (2,1)

tf.expand_dims(t,0) (1,2)

删除维度
只能删除长度为1的维度
t (1,2,1,3,1)
tf.squeeze(t) (2,3)
tf.squeeze(t,[2,4])  (1,2,3)
增加维度和删除维度 只是改变了张量的视图，不会改变张量的存储

交换维度
tf.transpose(a,perm) perm=[1,0]
[1,2,3],[4,5,6]  tf.transpose(x) [1,4],[2,5],[3,6]
(2,3,4)  tf.transpose(x,(1,0,2)) (3,2,4)
交换维度，不仅改变了张量的视图，同时也改变了张量的存储顺序

拼接和分割
拼接张量
tf.concat(tensors,axis)
(2,3)(2,3) 0 (4,3) 1(2,6)

分割张量 将一个张量拆分成多个张量，分割后维度不变
tf.split(value,num_or_size_splits,axis=0)
分割成2个张量
[1:2:1]就表示分割成3个张量 长度分别是1，2，1
[4,6]
tf.split(x,2,0) [2,6] [2,6]
图像的分割和拼接，改变了张量的视图，张量的存储顺序并没有改变

堆叠和分解
堆叠张量 创建一个新的维度
tf.stack(values,axis)

张量分解为多个张量
tf.unstack(values,axis)
"""