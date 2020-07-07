"""
end-to-end 端到端
open source 开放源代码 开放设计和实现框架
machine learning 机器学习生态系统

发展历程
2011： DistBelief
→2015.11: TensorFlow0.50
→2017.2: TensorFlow1.0
    高层 API 将 Keras 库整合进其中
    Eager Execution、TensorFlow Lite、TensorFlow.js
    AutoGraph
→2019:TensorFlow 2.0

TensorFlow2.0 特性

*TensorFlow1.x----延迟执行机制（deferred execution）
    构建阶段
    静态图只需要创建一次，可以重复使用
    静态图运行之前，可以优化，效率更高
    a=tf.constant(2,name='input_a')
    b=tf.constant(3,name='input_b')
    c=tf.add(a,b,name='add_c')
    执行阶段
    运行计算图
    sess=tf.Session()
    print(sess.run(c))
    sess.close()
    代码运行效率高 便于优化
    程序不够简洁
    TensorFlow2.0--动态图机制（Eager execution）
    a=tf.constant(2,name='input_a')
    b=tf.constant(3,name='input_b')
    print(a+b)
    无需首先创建静态图，可以立刻执行计算，并返回结果
    能够快速的建立和调试模型
    执行效率不高
    保留了静态图机制，我们可以在程序调试阶段使用动态图，快速建立模型、调试程序，在部署阶段，采用静态图机制，从而提高模型的性能和部署能力

* TensorFlow1.x 重复、冗余的 API
    构建神经网络:tf.slim tf.layers tf.contrib.layers tf.keras
    混乱 不利于程序共享，维护成本高
    TensorFlow2.0---清理/整合 API
    清理、整合了重复的 API
    将 tf.keras作为构建和训练模型的标准高级 API

*TensorFlow 框架特性
    多种环境支持
        可运行于移动设备、个人计算机、服务器、集群等
        云端、本地、浏览器、移动设备、嵌入式设备
    支持分布式模式
        TensorFlow 会自动检测 GPU 和 CPU 并充分利用它们并行、分布的执行
    简洁高效
        构建、训练、迭代模型：Eager Execution Keras
        部署阶段：转化为静态图，提高执行效率
    社区支持

"""