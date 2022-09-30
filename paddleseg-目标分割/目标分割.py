from paddleseg.models import BiSeNetV2
model = BiSeNetV2(num_classes=2,
                 lambd=0.25,
                 align_corners=False,
                 pretrained=None)


# 构建训练用的数据增强和预处理
import paddleseg.transforms as T
transforms = [
    T.Resize(target_size=(512, 512)),
    T.RandomHorizontalFlip(),
    T.Normalize()
]

# 构建训练集
from paddleseg.datasets import OpticDiscSeg
train_dataset = OpticDiscSeg(
    dataset_root='data/optic_disc_seg',
    transforms=transforms,
    mode='train'
)

