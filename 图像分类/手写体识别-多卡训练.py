# 加载相关库
import os
import random
import paddle
from paddle.nn import Conv2D, MaxPool2D, Linear
import numpy as np
from PIL import Image
import gzip
import json


# 定义数据集读取器
def load_data(mode='train'):
    # 读取数据文件
    datafile = './work/mnist.json.gz'
    print('loading mnist dataset from {} ......'.format(datafile))
    data = json.load(gzip.open(datafile))
    # 读取数据集中的训练集，验证集和测试集
    train_set, val_set, eval_set = data

    # 数据集相关参数，图片高度IMG_ROWS, 图片宽度IMG_COLS
    IMG_ROWS = 28
    IMG_COLS = 28
    # 根据输入mode参数决定使用训练集，验证集还是测试
    if mode == 'train':
        imgs = train_set[0]
        labels = train_set[1]
    elif mode == 'valid':
        imgs = val_set[0]
        labels = val_set[1]
    elif mode == 'eval':
        imgs = eval_set[0]
        labels = eval_set[1]
    # 获得所有图像的数量
    imgs_length = len(imgs)
    # 验证图像数量和标签数量是否一致
    assert len(imgs) == len(labels), \
        "length of train_imgs({}) should be the same as train_labels({})".format(
            len(imgs), len(labels))

    index_list = list(range(imgs_length))

    # 读入数据时用到的batchsize
    BATCHSIZE = 100

    # 定义数据生成器
    def data_generator():
        # 训练模式下，打乱训练数据
        if mode == 'train':
            random.shuffle(index_list)
        imgs_list = []
        labels_list = []
        # 按照索引读取数据
        for i in index_list:
            # 读取图像和标签，转换其尺寸和类型
            img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
            label = np.reshape(labels[i], [1]).astype('int64')
            imgs_list.append(img)
            labels_list.append(label)
            # 如果当前数据缓存达到了batch size，就返回一个批次数据
            if len(imgs_list) == BATCHSIZE:
                yield np.array(imgs_list), np.array(labels_list)
                # 清空数据缓存列表
                imgs_list = []
                labels_list = []

        # 如果剩余数据的数目小于BATCHSIZE，
        # 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
        if len(imgs_list) > 0:
            yield np.array(imgs_list), np.array(labels_list)

    return data_generator


# 定义模型结构
import paddle.nn.functional as F


# 多层卷积神经网络实现
class MNIST(paddle.nn.Layer):
    def __init__(self):
        super(MNIST, self).__init__()

        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv1 = Conv2D(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool1 = MaxPool2D(kernel_size=2, stride=2)
        # 定义卷积层，输出特征通道out_channels设置为20，卷积核的大小kernel_size为5，卷积步长stride=1，padding=2
        self.conv2 = Conv2D(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=2)
        # 定义池化层，池化核的大小kernel_size为2，池化步长为2
        self.max_pool2 = MaxPool2D(kernel_size=2, stride=2)
        # 定义一层全连接层，输出维度是10
        self.fc = Linear(in_features=980, out_features=10)

    # 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
    # 卷积层激活函数使用Relu，全连接层激活函数使用softmax
    def forward(self, inputs):
        x = self.conv1(inputs)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = paddle.reshape(x, [x.shape[0], 980])
        x = self.fc(x)
        return x


# 仅优化算法的设置有所差别
def train(model):
    # 开启GPU
    use_gpu = False #True # 训练GPU、CPU选用
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
    model.train()
    # 调用加载数据的函数
    train_loader = load_data('train')

    # 设置不同初始学习率
    opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())

    EPOCH_NUM = 5
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            # 准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)

            # 前向计算的过程
            predicts = model(images)

            # 计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)

            # 每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

            # 后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    # 保存模型参数
    paddle.save(model.state_dict(), 'mnist_cpu.pdparams')


# 创建模型
model = MNIST()
# 启动训练过程
# train(model)

# 数据并行
# 数据并行与模型并行不同，数据并行每次读取多份数据，读取到的数据输入给多个设备（GPU）上的模型，每个设备上的模型是完全相同的，飞桨采用的就是这种方式。
import paddle
import paddle.distributed as dist

def train_multi_gpu(model):
    # 修改1- 初始化并行环境
    dist.init_parallel_env()
    # 修改2- 增加paddle.DataParallel封装
    model = paddle.DataParallel(model)
    model.train()
    #调用加载数据的函数
    train_loader = load_data('train')
    opt = paddle.optimizer.Adam(learning_rate=0.01, parameters=model.parameters())
    EPOCH_NUM = 3
    for epoch_id in range(EPOCH_NUM):
        for batch_id, data in enumerate(train_loader()):
            #准备数据，变得更加简洁
            images, labels = data
            images = paddle.to_tensor(images)
            labels = paddle.to_tensor(labels)
            #前向计算的过程
            predicts = model(images)
            #计算损失，取一个批次样本损失的平均值
            loss = F.cross_entropy(predicts, labels)
            avg_loss = paddle.mean(loss)
            #每训练了100批次的数据，打印下当前Loss的情况
            if batch_id % 200 == 0:
                print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))
            #后向传播，更新参数的过程
            avg_loss.backward()
            opt.step()
            opt.clear_grad()

    #保存模型参数
    paddle.save(model.state_dict(), 'mnist_mul_data.pdparams')

paddle.set_device('cpu') #paddle.set_device('gpu')
#创建模型
model = MNIST()
#启动训练过程
train_multi_gpu(model)

#启动多GPU的训练，有两种方式：
#  基于launch启动；
#  基于spawn方式启动。
'''
1. 基于launch方式启动
需要在命令行中设置参数变量。打开终端，运行如下命令：

单机单卡启动，默认使用第0号卡。
$ python train.py
单机多卡启动，默认使用当前可见的所有卡。
$ python -m paddle.distributed.launch train.py
单机多卡启动，设置当前使用的第0号和第1号卡。
$ python -m paddle.distributed.launch --gpus '0,1' --log_dir ./mylog train.py
$ export CUDA_VISIABLE_DEVICES='0,1'
$ python -m paddle.distributed.launch train.py
相关参数含义如下：

paddle.distributed.launch：启动分布式运行。
gpus：设置使用的GPU的序号（需要是多GPU卡的机器，通过命令watch nvidia-smi查看GPU的序号）。
log_dir：存放训练的log，若不设置，每个GPU上的训练信息都会打印到屏幕。
train.py：多GPU训练的程序，包含修改过的train_multi_gpu()函数。
训练完成后，在指定的./mylog文件夹下会产生四个日志文件，其中worklog.0的内容如下：




2. 基于spawn方式启动
launch方式启动训练，是以文件为单位启动多进程，需要用户在启动时调用paddle.distributed.launch，对于进程的管理要求较高；飞桨最新版本中，增加了spawn启动方式，可以更好地控制进程，在日志打印、训练和退出时更加友好。spawn方式和launch方式仅在启动上有所区别。

# 启动train多进程训练，默认使用所有可见的GPU卡。
if __name__ == '__main__':
    dist.spawn(train)

# 启动train函数2个进程训练，默认使用当前可见的前2张卡。
if __name__ == '__main__':
    dist.spawn(train, nprocs=2)

# 启动train函数2个进程训练，默认使用第4号和第5号卡。
if __name__ == '__main__':
    dist.spawn(train, nprocs=2, selelcted_gpus='4,5')
'''



