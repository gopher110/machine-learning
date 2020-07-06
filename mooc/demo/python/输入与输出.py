# 判断变量 num 是否是正数
num = int(input('请输入一个整数：'))
print(f'您输入的整数是：{num}')
if num > 0:
    print('num 是正数')
else:
    print(r'num 可能是0 \ num 也可能是负数', end='\n 33')