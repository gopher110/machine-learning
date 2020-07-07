"""
Pillow
导入 PIL.image 模块

图像对象的主要属性
图像对象.format  图像格式
图像对象.size 图像尺寸
图像对象.mode 色彩模式

显示图像
imshow()
plt.imshow(image 对象/numpy 数组) cmap='gray'表示各个通道的图像以灰度图显示

图像对象.convert(色彩模式)

转换图像的色彩模式
取值      色彩模式
1       二值图像
L       灰度图像
P       8位彩色图像
RGB     24位彩色图像
RGBA    32位彩色图像
CMYK    CMYK 彩色图像
YCbCr   YCbCr 彩色图像
I       32位整型灰度图像
F       32位浮点灰度图像

颜色通道的分离与合并
图像对象.split()
Image.merge(色彩模式，图像列表)

转化为数组
np.array[图像对象]

对图像的缩放、旋转和镜像
缩放图像
图像对象.resize((width,height))
图像对象.thumbnail((width,height))原地操作，返回值 None

旋转、镜像
图像对象.transpose(旋转方式)
Image.FLIP_LEFT_RIGHT 水平翻转
Image.FLIP_TOP_BOTTOM 上下翻转
Image.ROTATE_90 逆时针旋转90度
Image.ROTATE_180 逆时针旋转180度
Image.ROTATE_270 逆时针旋转270度
Image.TRANSPOSE 将图像进行转置
Image.TRANSVERSE 将图像进行转置，再水平翻转


裁剪图像
在图像上指定位置裁剪出一个矩形区域
图像对象.crop((x0,y0,x1,y1))

"""
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
img = Image.open('lena.tiff')
img.save('lena.jpg')
img.save('lena.bmp')

arr_img = np.array(img)
print(f"shape:{arr_img.shape}\n {arr_img}")

img1 = Image.open('lena.jpg')
img2 = Image.open('lena.bmp')
print("image:", img.mode)
print("image1:", img1.format)
plt.figure(figsize=(30, 5))
img_r, img_g, img_b = img.split()

img_rgb = Image.merge('RGB', [img_r, img_g, img_b])

imgList = [img.transpose(Image.TRANSPOSE), img1.transpose(Image.ROTATE_180).crop((100, 100, 400, 400)),
           img2.transpose(Image.TRANSPOSE), img_r, img_g, img_b, img_rgb]
for i in range(7):
    img = imgList[i]
    plt.subplot(1, 7, i+1)
    plt.axis('off')  # 不显示坐标轴
    if i > 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    plt.title(img.format)

plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()

img_gray = img.convert('L')
print('mode: ', img_gray.mode)

img_gray.save('lena_gray.bmp')

# 将灰度图像 lena_gray.bmp 转化为数组
img_gray = Image.open('lena_gray.bmp')
arr_img_gray = np.array(img_gray)
print(f"gray shape:{arr_img_gray.shape}\n {arr_img_gray}")

arr_img_new = 255 - arr_img_gray  # 对图象中的每一个像素做反色处理
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.axis('off')
plt.imshow(arr_img_gray, cmap='gray')

plt.subplot(122)
plt.axis('off')
plt.imshow(arr_img_new, cmap='gray')

plt.show()