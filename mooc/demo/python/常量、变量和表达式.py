# python 支持 6种标准数据类型
# 数据、字符串、 列表、元组、字典、集合
"""
数据类型

数字（Numbers）
整型（Integer） 正整数 负整数 零
浮点数（Float） 小数 -0.28 1.392E9
布尔值(Boolean) True False
复数 (Complex) 1+2i

字符串（String）

标识符（Identifier）变量、函数、数组、文件、对象等的名字
标识符的第一个字符 必须是字母或者下划线
其他字符可以由字母下划线或者数字组成
标识符长度任意 大小写敏感
_xxx # 类中的保护变量名
__xxx # 类中的私有变量名
__xxx___ # 系统定义的标识符 __init__()
不能与关键字重名  help("keywords")

常量（constant）
数字 字符串 布尔值 空值
没有命名常量

变量（variable）
不需要声明
其数据类型由所赋的值来决定
不同类型的数字型数据运算时，会自动进行类型转化 bool<int<float<complex
print(1+True)   2
a=10
b=3.5
print(a+b) 13.5


int()
float()
bool()
str()
chr() chr(65) A
ord() ord('A') 65
"""


"""
运算符（Operator）：完成不同类型的常量、变量之间的运算
表达式（Expression）：由常量变量和运算符组成

算术运算符
+ - * / 
% 取模
** 幂运算
// 整除运算

赋值运算符
=
+=
-=
*=
/=
%=
**=
//=

连续赋值 
a=b=c=1

多元赋值
x,y=1,2
x,y=y,x

逻辑运算符
and 
or
not

比较运算符
==
!=,<>
<=
>=
<
>
3<4<5 连续比较

位运算符
& 按位与
| 按位或
^ 按位异或
~ 按位非
<< 位左移
>> 位右移

字符串运算符
+ 字符串连接
* 重复输出字符串
print("-"*40)
print(" "*15 + "这是分割线")
print("-"*40)

成员运算符
in
not in
str ='Hello'
'h' in str

"""
