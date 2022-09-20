import paddle
from paddle.vision.models import resnet50

# 调用高层API的resnet50模型
model = resnet50()
# 设置pretrained参数为True，可以加载resnet50在imagenet数据集上的预训练模型
# model = resnet50(pretrained=True)

# 随机生成一个输入
x = paddle.rand([1, 3, 224, 224])
# 得到残差50的计算结果
out = model(x)
# 打印输出的形状，由于resnet50默认的是1000分类
# 所以输出shape是[1x1000]
print(out.shape,out)

# 从paddle.vision.models 模块中import 残差网络，VGG网络，LeNet网络
from paddle.vision.models import resnet50, vgg16, LeNet
from paddle.vision.datasets import Cifar10
from paddle.optimizer import Momentum
from paddle.regularizer import L2Decay
from paddle.nn import CrossEntropyLoss
from paddle.metric import Accuracy
from paddle.vision.transforms import Transpose

# 确保从paddle.vision.datasets.Cifar10中加载的图像数据是np.ndarray类型
paddle.vision.set_image_backend('cv2')
# 调用resnet50模型
model = paddle.Model(resnet50(pretrained=False, num_classes=10))

# 使用Cifar10数据集
train_dataset = Cifar10(mode='train', transform=Transpose())
val_dataset = Cifar10(mode='test', transform=Transpose())
# 定义优化器
optimizer = Momentum(learning_rate=0.01,
                     momentum=0.9,
                     weight_decay=L2Decay(1e-4),
                     parameters=model.parameters())
# 进行训练前准备
model.prepare(optimizer, CrossEntropyLoss(), Accuracy(topk=(1, 5)))
# 启动训练
model.fit(train_dataset,
          val_dataset,
          epochs=1,
          batch_size=64,
          save_dir="./output",
          num_workers=8)

# training = False,直接导出 inference推理模型(静态模型)，否则导出的动态模型
model.save('model', training=False)  #training 设置为 False 表示训练完成。执行后，会导出 model.pdmodel、model.pdiparams.info、model.pdparams 三个文件。