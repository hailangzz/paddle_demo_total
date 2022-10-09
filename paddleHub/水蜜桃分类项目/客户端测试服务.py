import requests
import json
import cv2
import base64

import numpy as np


def cv2_to_base64(image):
    data = cv2.imencode('.jpg', image)[1]
    return base64.b64encode(data.tostring()).decode('utf8')

def base64_to_cv2(b64str):
    data = base64.b64decode(b64str.encode('utf8'))
    data = np.fromstring(data, np.uint8)
    data = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return data

# 发送HTTP请求
org_im = cv2.imread(r'E:\data\peach_data\test\M2\0.jpg')

data = {'images':[cv2_to_base64(org_im)], 'top_k':1}
headers = {"Content-type": "application/json"}
url = "http://10.20.8.31:8868/predict/resnet50_vd_imagenet_ssld"
print(data)
r = requests.post(url=url, headers=headers, data=json.dumps(data))
print(r)
data =r.json()["results"]['data']
print(data)