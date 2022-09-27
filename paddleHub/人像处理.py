# 人像扣图
import paddlehub as hub
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

module = hub.Module(name="deeplabv3p_xception65_humanseg")
res = module.segmentation(paths = ["./test.png"], visualization=True, output_dir='humanseg_output')


res_img_path = 'humanseg_output/test.png'
img = mpimg.imread(res_img_path)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()


# 人体部位分割
import paddlehub as hub
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

module = hub.Module(name="ace2p")
res = module.segmentation(paths = ["./test.png"], visualization=True, output_dir='ace2p_output')

res_img_path = './ace2p_output/test.png'
img = mpimg.imread(res_img_path)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()


# 人脸检测
import paddlehub as hub
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

module = hub.Module(name="ultra_light_fast_generic_face_detector_1mb_640")
res = module.face_detection(paths = ["./test.png"], visualization=True, output_dir='face_detection_output')

res_img_path = './face_detection_output/test.png'
img = mpimg.imread(res_img_path)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()

# 关键点检测
import paddlehub as hub
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

module = hub.Module(name="human_pose_estimation_resnet50_mpii")
res = module.keypoint_detection(paths = ["./test.png"], visualization=True, output_dir='keypoint_output')

res_img_path = './keypoint_output/test.jpg'
img = mpimg.imread(res_img_path)
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()