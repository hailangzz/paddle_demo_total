import os
import numpy as np

file1 = open("C:\\Users\\34426\\Documents\\WeChat Files\\wxid_gr1sr01q93ug22\\FileStorage\\File\\2022-09\\concat\\concat\\data\\in0_test.bin","rb")
file2 = open("C:\\Users\\34426\\Documents\\WeChat Files\\wxid_gr1sr01q93ug22\\FileStorage\\File\\2022-09\\concat\\concat\\data\\apic27858.bin","rb")

# 使用numpy保存“.bin”格式的矩阵，在读取时，是根据字节数来读取数据的：
data = np.memmap(file2, dtype='uint8', mode='r',)  # data = np.memmap(file1, dtype='float32', mode='r',)
print(data.shape,data)