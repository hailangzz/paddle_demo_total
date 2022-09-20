from visualdl import LogWriter

if __name__ == '__main__':
    value = [i/1000.0 for i in range(1000)]
    # 步骤一：创建父文件夹：log与子文件夹：scalar_test
    with LogWriter(logdir="./log/scalar_test") as writer:
        for step in range(1000):
            # 步骤二：向记录器添加一个tag为`train/acc`的数据
            writer.add_scalar(tag="train/acc", step=step, value=value[step])
            # 步骤二：向记录器添加一个tag为`train/loss`的数据
            writer.add_scalar(tag="train/loss", step=step, value=1/(value[step] + 1))
    # 步骤一：创建第二个子文件夹scalar_test2
    value = [i/500.0 for i in range(1000)]
    with LogWriter(logdir="./log/scalar_test2") as writer:
        for step in range(1000):
            # 步骤二：在同样名为`train/acc`下添加scalar_test2的accuracy的数据
            writer.add_scalar(tag="train/acc", step=step, value=value[step])
            # 步骤二：在同样名为`train/loss`下添加scalar_test2的loss的数据
            writer.add_scalar(tag="train/loss", step=step, value=1/(value[step] + 1))
