'''
1.安装Paddle Serving
其中客户端安装包支持Centos 7和Ubuntu 18，或者您可以使用HTTP服务，这种情况下不需要安装客户端。

!pip install paddle-serving-client==0.6
!pip install paddle-serving-server==0.6

2.保存模型
由于Paddle Serving部署一般需要额外的配置文件，所以Paddle Serving提供了一个save_model的API接口用于保存模型，该接口与save_inference_model类似，但是可将Paddle Serving在部署阶段需要用到的参数与配置文件统一保存打包。

import paddle_serving_client.io as serving_io
serving_io.save_model("housing_model", "housing_client_conf",
                      {"words": x}, {"prediction": y_predict},
                      paddle.static.default_main_program())


paddle.static.default_main_program()是静态图中执行图的概念，如果是动态图的模式编写程序，建议采用接下来讲的方案。

如果已使用save_inference_model接口保存好模型，Paddle Serving也提供了inference_model_to_serving接口，该接口可以把已保存的模型转换成可用于Paddle Serving使用的模型文件。

import paddle_serving_client.io as serving_io
serving_io.inference_model_to_serving(dirname=path, serving_server="serving_model", serving_client="client_conf",  model_filename=None, params_filename=None)
python -m paddle_serving_client.convert --dirname ./your_inference_model_dir

'''

# 使用paddle_serving_client.io实现对房价预测模型的保存
import paddle
from paddle.nn import Linear
import paddle.nn.functional as F
import numpy as np
import os
import random


def load_data():
    # 从文件导入数据
    datafile = r'D:\PycharmProgram\paddle_demo_total\波士顿房价预估\work//housing.data'
    data = np.fromfile(datafile, sep=' ', dtype=np.float32)

    # 每条数据包括14项，其中前面13项是影响因素，第14项是相应的房屋价格中位数
    feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', \
                     'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
    feature_num = len(feature_names)

    # 将原始数据进行Reshape，变成[N, 14]这样的形状
    data = data.reshape([data.shape[0] // feature_num, feature_num])

    # 将原数据集拆分成训练集和测试集
    # 这里使用80%的数据做训练，20%的数据做测试
    # 测试集和训练集必须是没有交集的
    ratio = 0.8
    offset = int(data.shape[0] * ratio)
    training_data = data[:offset]

    # 计算train数据集的最大值，最小值，平均值
    maximums, minimums, avgs = training_data.max(axis=0), training_data.min(axis=0), \
                               training_data.sum(axis=0) / training_data.shape[0]

    # 记录数据的归一化参数，在预测时对数据做归一化
    global max_values
    global min_values
    global avg_values
    max_values = maximums
    min_values = minimums
    avg_values = avgs

    # 对数据进行归一化处理
    for i in range(feature_num):
        data[:, i] = (data[:, i] - avgs[i]) / (maximums[i] - minimums[i])

    # 训练集和测试集的划分比例
    training_data = data[:offset]
    test_data = data[offset:]

    return training_data, test_data


class Regressor(paddle.nn.Layer):

    # self代表类的实例自身
    def __init__(self):
        # 初始化父类中的一些参数
        super(Regressor, self).__init__()

        # 定义一层全连接层，输入维度是13，输出维度是1
        self.fc = Linear(in_features=13, out_features=1)

    # 网络的前向计算
    @paddle.jit.to_static
    def forward(self, inputs):
        print(inputs.shape)
        x = self.fc(inputs)
        return x


# 声明定义好的线性回归模型
model = Regressor()
# 开启模型训练模式
model.train()
# 加载数据
training_data, test_data = load_data()
print("train data", len(training_data), len(training_data[0]))
# 定义优化算法，使用随机梯度下降SGD
# 学习率设置为0.01
opt = paddle.optimizer.SGD(learning_rate=0.01, parameters=model.parameters())

EPOCH_NUM = 10  # 设置外层循环次数
BATCH_SIZE = 10  # 设置batch大小

# 定义外层循环
for epoch_id in range(EPOCH_NUM):
    # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
    np.random.shuffle(training_data)
    # 将训练数据进行拆分，每个batch包含10条数据
    mini_batches = [training_data[k:k + BATCH_SIZE] for k in range(0, len(training_data), BATCH_SIZE)]
    # 定义内层循环
    for iter_id, mini_batch in enumerate(mini_batches):
        x = np.array(mini_batch[:, :-1])  # 获得当前批次训练数据
        y = np.array(mini_batch[:, -1:])  # 获得当前批次训练标签（真实房价）
        # 将numpy数据转为飞桨动态图tensor形式
        house_features = paddle.to_tensor(x)
        prices = paddle.to_tensor(y)
        # 前向计算
        predicts = model(house_features)
        # 计算损失
        loss = F.square_error_cost(predicts, label=prices)
        avg_loss = paddle.mean(loss)
        if iter_id % 20 == 0:
            print("epoch: {}, iter: {}, loss is: {}".format(epoch_id, iter_id, avg_loss.numpy()))
        # 反向传播
        avg_loss.backward()
        # 最小化loss,更新参数
        opt.step()
        # 清除梯度
        opt.clear_grad()

'''
保存serving格式模型
import paddle_serving_client
paddle_serving_client.io.save_dygraph_model("uci_housing_server", "uci_housing_client", model)


# 启动paddle serving 服务
# 此段代码在AI Studio上运行无法停止，需要手动中止再运行下面的部分
# 可以在本地上后台运行
!python -m paddle_serving_server.serve --model uci_housing_model --thread 10 --port 9292 --name uci

# 访问paddle serving 服务：
curl -H "Content-Type:application/json" -X POST -d '{"feed":[{"x": [0.0137, -0.1136, 0.2553, -0.0692, 0.0582, -0.0727, -0.1583, -0.0584, 0.6283, 0.4919, 0.1856, 0.0795, -0.0332]}], "fetch":["price"]}' http://127.0.0.1:9292/uci/prediction
'''