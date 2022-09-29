from __future__ import print_function
import paddle
from paddle.vision.models import mobilenet_v1

import paddle.vision.transforms as T
transform = T.Compose([
                    T.Transpose(),
                    T.Normalize(mean=[127.5], std=[127.5],data_format='CHW')
                ])
train_dataset = paddle.vision.datasets.Cifar10(mode="train", backend="cv2",transform=transform)
val_dataset = paddle.vision.datasets.Cifar10(mode="test", backend="cv2",transform=transform)


# 1.模型定义
#使用预置的mobilenet模型，但不使用预训练的参数
net = mobilenet_v1(pretrained=False)
paddle.summary(net, (1, 3, 32, 32))

model = paddle.Model(net)
# 加载模型
model.load(r'D:\PycharmProgram\paddle_train_model\paddle_slim/mobilenet_v1_Cifar10')

optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1,
        parameters=net.parameters())

model.prepare(
        optimizer,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5))) # #解决 AttributeError: 'Model' object has no attribute '_place' 问题
# 裁剪

from paddleslim.dygraph import L1NormFilterPruner
pruner = L1NormFilterPruner(net, [1, 3, 224, 224])

# 敏感度计算
def eval_fn():
    result = model.evaluate(
        val_dataset)
    return result['acc_top1']
pruner.sensitive(eval_func=eval_fn, sen_file="./sen.pickle")


# 裁剪
from paddleslim.analysis import dygraph_flops
flops = dygraph_flops(net, [1, 3, 32, 32])
print(f"FLOPs before pruning: {flops}")


#执行剪裁操作，期望跳过最后一层卷积层并剪掉40%的FLOPs，skip_vars参数可以指定不期望裁剪的参数结构。
plan = pruner.sensitive_prune(0.4, skip_vars=["conv2d_26.w_0"])
flops = dygraph_flops(net, [1, 3, 32, 32])
print(f"FLOPs after pruning: {flops}")
print(f"Pruned FLOPs: {round(plan.pruned_flops*100, 2)}%")


#通常，剪裁之后，模型的精度会大幅下降。如下所示，在测试集上重新评估精度，精度大幅下降：

result = model.evaluate(val_dataset,batch_size=128, log_freq=10)
print(f"before fine-tuning: {result}")

# 因此，需要对剪裁后的模型重新训练, 从而提升模型的精度，精度的提升取决于模型的要求。我们再训练之后在测试集上再次测试精度，会发现精度提升如下：

optimizer = paddle.optimizer.Momentum(
        learning_rate=0.1,
        parameters=net.parameters())
model.prepare(
        optimizer,
        paddle.nn.CrossEntropyLoss(),
        paddle.metric.Accuracy(topk=(1, 5)))
model.fit(train_dataset, epochs=1, batch_size=128, verbose=1)
model.save('D:\PycharmProgram\paddle_train_model\paddle_slim/mobilenet_v1_Cifar10_slim')
result = model.evaluate(val_dataset,batch_size=128, log_freq=10)
print(f"after fine-tuning: {result}")

# 打印模型参数
paddle.summary(net, (1, 3, 32, 32))