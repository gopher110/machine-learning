"""
函数（function）
 系统函数 用户自定义函数
 abs() 返回绝对值
 pow(x,y) x的y 次幂
 round(x,n)返回浮点数四舍五入的值 n 指定保留的小数位 默认为0
 divmod(a,b) 返回元祖（a/b,a%b）

max()返回最大值
sum()返回总和
open()打开文件
range() 返回一个整型列表
type()显示类型
dir()显示属性
help()显示帮助信息
list()转化为列表

用户自定义函数
def 函数名（参数列表）：
    函数体


调用函数
函数名（实参）
通过多元赋值语句 同时获取多个返回值
def add_mul(a,b):
    add=a+b
    mul=a*b
    return add,mul

x,y=add_mul(1,2) 小括号不能省

def sum(list)
当使用列表或字典作为参数时，在函数内部若对列表或字典的元素进行修改，会改变实参的值
"""


"""
模块（module ）
模块是一个 py 文件 拥有多个功能相近的函数或类。
便于代码复用，提高编程效率，提高了代码的可维护性，避免
函数名和变量名冲突。

包（package）
为了避免模块名冲突，python 引入按目录来组织模块的方法。
一个包对应一个文件夹，将模块文件放在一个文件夹下。
作为包的文件夹下有一个__init__.py 文件。
子包： 子目录中也有__init__.py 文件。


导入模块或包
import numpy as np
np.random.random()

导入包中指定的模块或子包
from 模块/包名称 import 函数名 as 函数别名
from numpy import *
random.random()

创建自定义模块 mymodule.py
import mymodule

sys.platform:获取当前操作系统
sys.path:获取指定模块搜索路径

sys.append():添加指定模块搜索路径
sys.path.append()
"""