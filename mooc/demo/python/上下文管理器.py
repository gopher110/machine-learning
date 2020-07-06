"""
异常处理
try:
  f = open('mypython.txt')
  print(f.read())
except IOError as e:
    print(e)
finally:
    f.close()


with语句

with open('mypython.txt') as f:
    print(f.read())

上下文管理器

实现__enter__()和__exit__()方法


class File():
    def __init__(self,filename,mode):
        print('执行 init)
        self.filename = filename
        self.mode = mode
    def __enter__(self):
        self.f = open(self.filename, self.mode)
        return self.f

    def __exit__(self,*args):
        self.f.close()
"""