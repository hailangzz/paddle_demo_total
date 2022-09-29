import paddle
from paddle.autograd import PyLayer

# 通过创建`PyLayer`子类的方式实现动态图Python Op
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x):
        y = paddle.tanh(x)
        # ctx 为PyLayerContext对象，可以把y从forward传递到backward。
        ctx.save_for_backward(y)
        return y

    @staticmethod
    # 因为forward只有一个输出，因此除了ctx外，backward只有一个输入。
    def backward(ctx, dy):
        # ctx 为PyLayerContext对象，saved_tensor获取在forward时暂存的y。
        y, = ctx.saved_tensor()
        # 调用Paddle API自定义反向计算
        grad = dy * (1 - paddle.square(y))
        # forward只有一个Tensor输入，因此，backward只有一个输出。
        return grad

data = paddle.randn([2, 3], dtype="float32")
data.stop_gradient = False
# 通过 apply运行这个Python算子
z = cus_tanh.apply(data)
z.mean().backward()

print(data.grad)






import paddle
from paddle.autograd import PyLayer

# Inherit from PyLayer
class cus_tanh(PyLayer):
    @staticmethod
    def forward(ctx, x, func1, func2=paddle.square):
        # ctx is a context object that store some objects for backward.
        ctx.func = func2
        y = func1(x)
        # Pass tensors to backward.
        ctx.save_for_backward(y)
        return y

    @staticmethod
    # forward has only one output, so there is only one gradient in the input of backward.
    def backward(ctx, dy):
        # Get the tensors passed by forward.
        y, = ctx.saved_tensor()
        grad = dy * (1 - ctx.func(y))
        # forward has only one input, so only one gradient tensor is returned.
        return grad


# data = paddle.randn([2, 3], dtype="float64")

data.stop_gradient = False
z = cus_tanh.apply(data, func1=paddle.tanh)
z.mean().backward()

print(data.grad.numpy())