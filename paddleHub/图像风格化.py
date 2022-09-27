import  os

import cv2
import paddlehub as hub
import matplotlib.pyplot as plt
#%matplotlib inline

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model = hub.Module(name='animegan_v2_hayao_64', use_gpu=True)

# 模型预测
result = model.style_transfer(images=[cv2.imread('test.jpg')])
plt.figure(figsize=(10,10))
plt.imshow(result[0][:,:,[2,1,0]])
plt.show()