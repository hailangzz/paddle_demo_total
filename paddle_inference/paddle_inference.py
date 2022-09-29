# 导入Paddle相关包
import paddle
from paddle.jit import to_static
from paddle.static import InputSpec
from paddle.nn import Layer


# 1. tostatic裝飾器模式：
# 定义线性回归网络，继承自paddle.nn.Layer
# 该网络仅包含一层fc
class SimpleNet(Layer):
# 在__init__函数中仅初始化linear层
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

# 在forward函数中定义该网络的具体前向计算；@to_static装饰器用于依次指定参数x和y对应Tensor签名信息
# 下述案例是输入为10个特征，输出为1维的数字
    @to_static(input_spec=[InputSpec(shape=[None, 10], name='x'), InputSpec(shape=[1], name='y')])
    def forward(self, x, y):
        out = self.linear(x)
        out = out + y
        return out

net = SimpleNet()

# 保存预测格式模型
paddle.jit.save(net, './simple_net')


# 2. to_static函数调用模式： 若期望在动态图下训练模型，在训练完成后保存预测模型，并指定预测时需要的签名信息，则可以选择在保存模型时，直接调用 to_static 函数。使用样例如下：

# 定义线性回归网络，继承自paddle.nn.Layer
# 该网络仅包含一层fc
class SimpleNet(Layer):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear = paddle.nn.Linear(10, 3)

# 在forward函数中定义该网络的具体前向计算
    def forward(self, x, y, z):
        out = self.linear(x)
        out = out + y
        out = out * z
        return out

net = SimpleNet()

# 训练过程 (伪代码)
for epoch_id in range(10):
    pass
    #train_step(net, train_reader)

# 直接调用to_static函数，paddle会根据input_spec信息对forward函数进行递归的动转静，得到完整的静态图模型
net = to_static(net, input_spec=[InputSpec(shape=[1, 10], name='x'), InputSpec(shape=[3], name='y'), InputSpec(shape=[3], name='z')])

# 保存预测格式模型
paddle.jit.save(net, './simple_net2')

# ----------------------------------------------------------------------------------------------------------------------

# Paddle Inference Python 接口的部署
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer
# 引用 numpy argparse库
import numpy as np
import argparse

# 创建配置对象，并根据需求配置
config = paddle_infer.Config("simple_net.pdmodel", "simple_net.pdiparams")

# 配置config，不启用gpu
config.disable_gpu()

# 根据Config创建预测对象
predictor = paddle_infer.create_predictor(config)

# 获取输入的名称
input_names = 	predictor.get_input_names()
# 获取输入handle
x_handle = predictor.get_input_handle(input_names[0])
y_handle = predictor.get_input_handle(input_names[1])

# 设置输入，此处以随机输入为例，用户可自行输入真实数据
fake_x = np.random.randn(1, 10).astype('float32')
fake_y = np.random.randn(1).astype('float32')

print(fake_y)
x_handle.reshape([1, 10])
x_handle.copy_from_cpu(fake_x)
y_handle.reshape([1])
y_handle.copy_from_cpu(fake_y)

# 运行预测引擎
predictor.run()

# 获得输出名称
output_names = predictor.get_output_names()
# 获得输出handle
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu() # return numpy.ndarray

# 打印输出结果
print(output_data)



# Paddle Inference Python 接口的部署
# 引用 paddle inference 预测库
import paddle.inference as paddle_infer
# 引用 numpy argparse库
import numpy as np
import argparse

# 创建配置对象，并根据需求配置
config = paddle_infer.Config("simple_net2.pdmodel", "simple_net2.pdiparams")

# 配置config，不启用gpu
config.disable_gpu()

# 根据Config创建预测对象
predictor = paddle_infer.create_predictor(config)

# 获取输入的名称
input_names = 	predictor.get_input_names()
# 获取输入handle
x_handle = predictor.get_input_handle(input_names[0])
y_handle = predictor.get_input_handle(input_names[1])
z_handle = predictor.get_input_handle(input_names[2])

# 设置输入，此处以随机输入为例，用户可自行输入真实数据
fake_x = np.random.randn(1, 10).astype('float32')
fake_y = np.random.randn(3).astype('float32')
fake_z = np.random.randn(3).astype('float32')

print(fake_y)
print(fake_z)
x_handle.reshape([1, 10])
x_handle.copy_from_cpu(fake_x)
y_handle.reshape([3])
y_handle.copy_from_cpu(fake_y)
z_handle.reshape([3])
z_handle.copy_from_cpu(fake_z)

# 运行预测引擎
predictor.run()

# 获得输出名称
output_names = predictor.get_output_names()
# 获得输出handle
output_handle = predictor.get_output_handle(output_names[0])
output_data = output_handle.copy_to_cpu() # return numpy.ndarray

# 打印输出结果
print(output_data)