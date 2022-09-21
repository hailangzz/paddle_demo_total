import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image, ImageEnhance

image_path=r'D:\迅雷下载\AI数据集汇总\害虫检测数据集\data\data19638\insects\train\images\1.jpeg'

# coding=gbk
from PIL import Image
import numpy as np
# import scipy
import matplotlib.pyplot as plt

def ImageToMatrix(filename):
    # 读取图片
    im = Image.open(filename)
    # 显示图片
#     im.show()
    width,height = im.size
    im = im.convert("L")
    data = im.getdata()
    data = np.matrix(data,dtype='float')/255.0
    #new_data = np.reshape(data,(width,height))
    new_data = np.reshape(data,(height,width))
    return new_data
#     new_im = Image.fromarray(new_data)
#     # 显示图片
#     new_im.show()
def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im



filename = r'D:\迅雷下载\AI数据集汇总\害虫检测数据集\data\data19638\insects\train\images\1.jpeg'
data = ImageToMatrix(filename)
print(data)
new_im = MatrixToImage(data)
print(new_im)
plt.imshow(data, cmap=plt.cm.gray, interpolation='nearest')
new_im.show()
new_im.save('lena_1.bmp')