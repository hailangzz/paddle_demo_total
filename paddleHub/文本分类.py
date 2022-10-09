import paddlehub as hub
model = hub.Module(name='ernie_tiny', version='2.0.1', task='seq-cls', num_classes=2)

# 2. 准备数据集并读取数据
# 自动从网络下载数据集并解压到用户目录下$HUB_HOME/.paddlehub/dataset目录
train_dataset = hub.datasets.ChnSentiCorp(
    tokenizer=model.get_tokenizer(), max_seq_len=128, mode='train')
dev_dataset = hub.datasets.ChnSentiCorp(
    tokenizer=model.get_tokenizer(), max_seq_len=128, mode='dev')

# 3.自定义文本分类数据集
'''
from paddlehub.datasets.base_nlp_dataset import TextClassificationDataset
class SeqClsDataset(TextClassificationDataset):
    # 数据集存放目录
    base_path = '/path/to/dataset'
    # 数据集的标签列表
    label_list = ['体育', '科技', '社会', '娱乐', '股票', '房产', '教育', '时政', '财经', '星座', '游戏', '家居', '彩票', '时尚']

    def __init__(self, tokenizer, max_seq_len: int = 128, mode: str = 'train'):
        if mode == 'train':
            data_file = 'train.txt'
        elif mode == 'test':
            data_file = 'test.txt'
        else:
            data_file = 'dev.txt'
        super().__init__(
            base_path=self.base_path,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            mode=mode,
            data_file=data_file,
            label_list=self.label_list,
            is_file_with_header=True)


# 选择所需要的模型，获取对应的tokenizer
import paddlehub as hub

model = model = hub.Module(name='ernie_tiny', task='seq-cls', num_classes=len(SeqClsDataset.label_list))
tokenizer = model.get_tokenizer()

# 实例化训练集
train_dataset = SeqClsDataset(tokenizer)
'''


import paddle

optimizer = paddle.optimizer.Adam(learning_rate=5e-5, parameters=model.parameters())
trainer = hub.Trainer(model, optimizer, checkpoint_dir='test_ernie_text_cls', use_gpu=False)

trainer.train(train_dataset, epochs=3, batch_size=32, eval_dataset=dev_dataset, save_interval=1)