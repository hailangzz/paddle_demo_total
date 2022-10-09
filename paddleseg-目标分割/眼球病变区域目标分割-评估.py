import paddle
from paddleseg.models import BiSeNetV2
model = BiSeNetV2(num_classes=2,
                 lambd=0.25,
                 align_corners=False,
                 pretrained=None)

model_path = 'D:\PycharmProgram\paddle_train_model\paddle_seq\\best_model/model.pdparams'
if model_path:
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    print('Loaded trained params of model successfully')
else:
    raise ValueError('The model_path is wrong: {}'.format(model_path))


# 构建验证用的transforms
import paddleseg.transforms as T
transforms = [
    T.Resize(target_size=(512, 512)),
    T.Normalize()
]

# 构建验证集
from paddleseg.datasets import OpticDiscSeg
val_dataset = OpticDiscSeg(
    dataset_root='data/optic_disc_seg',
    transforms=transforms,
    mode='val'
)

from paddleseg.core import evaluate
evaluate(
        model,
        val_dataset)

#多尺度+翻转评估
evaluate(
        model,
        val_dataset,
        aug_eval=True,
        scales=[0.75, 1.0, 1.25],
        flip_horizontal=True)