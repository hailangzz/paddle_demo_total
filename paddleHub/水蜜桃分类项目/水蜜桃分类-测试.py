import paddle
import paddlehub as hub


model = hub.Module(name='resnet50_vd_imagenet_ssld', label_list=["R0", "B1", "M2", "S3"])
result = model.predict(['./work/peach-classification/test/M2/0.png'])
print(result)
