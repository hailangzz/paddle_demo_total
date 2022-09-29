from __future__ import print_function
import paddle
from paddle.vision.models import mobilenet_v1


# 1.模型定义
#使用预置的mobilenet模型，但不使用预训练的参数
net = mobilenet_v1(pretrained=False)
paddle.summary(net, (1, 3, 32, 32))

# 2.准备数据
# 我们直接使用vision模块提供的Cifar10 数据集，并通过飞桨高层API paddle.vision.transforms对数据进行预处理。在声明paddle.vision.datasets.Cifar10对象时，会自动下载数据并缓存到本地文件系统。代码如下所示：

import paddle.vision.transforms as T
transform = T.Compose([
                    T.Transpose(),
                    T.Normalize(mean=[127.5], std=[127.5],data_format='CHW')
                ])
train_dataset = paddle.vision.datasets.Cifar10(mode="train", backend="cv2",transform=transform)
val_dataset = paddle.vision.datasets.Cifar10(mode="test", backend="cv2",transform=transform)


print(f'train samples count: {len(train_dataset)}')
print(f'val samples count: {len(val_dataset)}')
for data in train_dataset:
    print(f'image shape: {data[0].shape}; label: {data[1]}')
    break
'''   
3. 模型训练准备工作
在对卷积网络进行剪裁之前，我们需要在测试集上评估网络中各层的重要性。在剪裁之后，我们需要对得到的小模型进行重训练。在本示例中，我们将会使用Paddle高层API paddle.Model进行训练和评估工作。以下代码声明了paddle.Model实例，并指定了训练相关的一些设置，包括：

输入的shape
优化器
损失函数
模型评估指标
'''

from paddle.static import InputSpec as Input
optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1,
        parameters=net.parameters())

inputs = [Input([None, 3, 32, 32], 'float32', name='image')]
labels = [Input([None], 'int64', name='label')]

model = paddle.Model(net, inputs, labels)

model.prepare(
        optimizer,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5)))

model.fit(train_dataset, epochs=1, batch_size=228, verbose=1)
# 保存模型
model.save(r'D:\PycharmProgram\paddle_train_model\paddle_slim/mobilenet_v1_Cifar10')
result = model.evaluate(val_dataset,batch_size=128, log_freq=10)
print(result)

'''
4. 剪裁
本节内容分为两部分：卷积层重要性分析和Filters剪裁，其中『卷积层重要性分析』也可以被称作『卷积层敏感度分析』，我们定义越重要的卷积层越敏感。
敏感度的理论计算过程如下图所示：第一层卷积操作有四个卷积核，首先计算每个卷积核参数的L1_norm值，即所有参数的绝对值之和。之后按照每个卷积的L1_norm值排序，
先去掉L1_norm值最小的（即图中L1_norm=1的卷积核），测试模型的效果变化，再去掉次小的（即图中L1_norm=1.2的卷积核），测试模型的效果变换，以此类推。
观察每次裁剪的模型效果曲线绘图，那些裁剪后模型效果衰减不显著的卷积核会被删除。
因此，敏感度通俗的理解就是每个卷积核对最终预测结果的贡献度或者有效性，那些对最终结果影响不大的部分会被裁掉。
'''
from paddleslim.dygraph import L1NormFilterPruner
pruner = L1NormFilterPruner(net, [1, 3, 224, 224])


#4.1 敏感度计算
# 调用pruner对象的sensitive方法进行敏感度分析，在调用sensitive之前，我们简单对model.evaluate进行包装，使其符合sensitive接口的规范。执行如下代码，会进行敏感度计算，并将计算结果存入本地文件系统：

def eval_fn():
    result = model.evaluate(
        val_dataset,
        batch_size=128)
    return result['acc_top1']
pruner_info = pruner.sensitive(eval_func=eval_fn, sen_file="./sen.pickle")
print(pruner_info)










