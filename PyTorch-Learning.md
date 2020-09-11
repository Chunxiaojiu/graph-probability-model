# PyTorch学习手册
随机学习其中函数，随缘记录，未来整理！

----
## Module `torch.nn.Module`
该类作为所有神经网络模块的基类，用户的模型也需要继承该类。这个模块可以包含其他的模块，也可以将他们嵌套在一个类似树形的结构中，我们接下来看看她的常规属性的分配：
``` Python

import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x)) 
```
> `add_module(name: str, module: torch.nn.modules.module.Module) → None`
<br>
添加一个子模型，这个模型可以通过给定的名称访问。

>>> 参数介绍
>>* name(string) -> 子模型的名字，同时可以通过该名称直接使用这个模型。
>>* module(Moudule) -> 将要加入该模型的子模型。


>`apply(fn: Callable[Module, None]) → T` 
>>将$f_{n}$递归的使用在所有的子模块中（通过`.children()`进行返回的模块），典型用途包括初始化模型的参数。
>>>参数介绍
>>> + **$fn$**(Moudle -> None) 作为适用于所有子模型中的函数


>>>返回值
>>> + **$self$**


>>>返回类型
>>> + **$Module$**


## `load_state_dict(state_dict: Dict[str, torch.Tensor], strict: bool = True)`
将参数和缓冲区从state_dict复制到此模块及其后代中。如果strict为True，则state_dict的键必须与该模块的state_dict（）函数返回的键完全匹配。

## `torch.no_grad`
禁用梯度计算的上下文管理器。当你确定你将不会调用`Tensor.backward()`时，禁用梯度计算对于推断非常有效。当我们不将*`requires_gard = True`*它将减少用于计算的内存消耗。在当前模式下所有的计算将会认定为不进行梯度计算，即使在输入的时候标记为进行梯度计算。当使用 **`enable_grad`**时该模式没有作用。同时该模式只基于当前线程，其他线程不起作用。
``` python
>>> with torch.no_grad():
>>> @torch.no_grad()
False
```

## `torch.nn.Dropout(p: float = 0.5, inplace: bool = False)`
在训练期间，使用伯努利分布的样本以概率p将输入张量的某些元素随机置零。在每个前向调用中，每个通道将独立清零。事实证明，这是一种有效的技术，可用于规范化和防止神经元的协同适应，在论文Improving neural networks by preventing co-adaptation of feature detectors[^1]中进行论证。
此外，输出将会按照$\frac{1}{1-p}$来进行缩放。这意味着在计算期间，模块仅计算身份函数。

[^1]:https://arxiv.org/abs/1207.0580

## `torch.nn.Embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None, max_norm: Optional[float] = None, norm_type: float = 2.0, scale_grad_by_freq: bool = False, sparse: bool = False, _weight: Optional[torch.Tensor] = None)`
一个简单的查找表，用于存储固定字典和大小的嵌入。该模块通常用于存储单词嵌入并使用索引检索它们。模块的输入是索引列表，而输出是相应的词嵌入。
>>参数介绍
>> + `num_embedding(int)`嵌入词典的大小
>> + `embedding_dim (int)`每个嵌入向量的大小 
>> + `padding_idx (int, optional)`如果给定，则在遇到索引时在`padding_idx（初始化为零）`处用嵌入矢量填充输出
>> + `max_norm (float, optional) `将范数大于`max_norm`的每个嵌入向量重新规范化为范数`max_norm`
>> + `norm_type (float, optional)`为`max_norm`选项指定的p范数，默认为2
>> + `scale_grad_by_freq (boolean, optional) `如果给定的话，这将按小批量中单词频率的倒数来缩放梯度
>> + `sparse (bool, optional) `梯度权重矩阵将会是一个稀疏张量


```python 
>>> # an Embedding module containing 10 tensors of size 3
>>> embedding = nn.Embedding(10, 3)
>>> # a batch of 2 samples of 4 indices each
>>> input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
>>> embedding(input)
tensor([[[-0.0251, -1.6902,  0.7172],
         [-0.6431,  0.0748,  0.6969], //注意10个长度为3的张量字典 tensor—>2
         [ 1.4970,  1.3448, -0.9685],
         [-0.3677, -2.7265, -0.1685]],

        [[ 1.4970,  1.3448, -0.9685],
         [ 0.4362, -0.4004,  0.9400],
         [-0.6431,  0.0748,  0.6969], // tensor—>2
         [ 0.9124, -2.3616,  1.1151]]])


>>> # example with padding_idx
>>> embedding = nn.Embedding(10, 3, padding_idx=0)
>>> input = torch.LongTensor([[0,2,0,5]])
>>> embedding(input)
tensor([[[ 0.0000,  0.0000,  0.0000],
         [ 0.1535, -2.0309,  0.9315],
         [ 0.0000,  0.0000,  0.0000],
         [-0.1655,  0.9897,  0.0635]]])
```

## LSTM `torch.nn.LSTM(*args, **kwargs)`
将多层长短期记忆（LSTM）RNN应用于输入序列。
对于输入序列中的每个元素，每个层都会计算以下函数：

<center> 

$i_t = \sigma(W_{ii}x_t + b_{ii}+W_{ii}h_{t-1}+b_{hi})$

$f_t = \sigma(W_{if}x_t + b_{if}+W_{hf}h_{t-1}+b_{hf})$

$g_t = tanh(W_{ig}x_t + b_{io} +W_{hg}h_{t-1} +b_{hg})$

$o_t = \sigma(W_{io}x_t + b_{io}+W_{ho}h_{t-1}+b_{ho})$

$c_t = f_t\odot c_{t-1} + i_t\odot g_t$

$h_t = o_t \odot tanh(c_t)$
</center>

其中，$h_t$ 表示在时间$t$时的隐藏状态，$c_t$表示$t$时刻单元状态，$x_t$表示在$t$时刻的输入，$h_{t-1}$表示该层隐藏状态在$t-1$时刻或者初始隐藏状态在时间$o$的状态，然后$i_t,f_t,g_t,o_t$分别时输入，忘记，单元和输出门。$\sigma$表示$sigmod$函数，然后$\odot$表示Hadamard乘积。
在一个多重的LSTM网络中，第$l-th$层的输入$x^{l}_t$是前一层的隐藏状态$h^(l-1)_t$乘以遗忘值$\delta _{t} ^{l-1}$,其中$\delta _{t} ^{l-1}$是一个*Bernouli* 随机变量取值为0概率为`dropout`。

>>参数介绍
>> - input_size 对于输入x预期的特征数目
>> - hidden_size 隐藏状态h的特征数目
>> - num_layers 递归层数。例子，设定该值为2，意思着这意味着将两个LSTM堆叠在一起以形成一个堆叠的LSTM，而第二个LSTM则接收第一个LSTM的输出并计算最终结果。
>> - bias 如果设定为`false` ,将不会使用偏置权重$b_{ih}$和$b_{hh}$。

## `torch.nn.Linear(in_features: int, out_features: int, bias: bool = True)`
对于输入的数据实现一个线性变化：$y = xA^T + b$


>> 参数介绍
>> + in_features 输入数据的大小
>> + out_features 输出数据的大小
>> + bias 是否使用权重

```python 
>>> m = nn.Linear(20, 30)
>>> input = torch.randn(128, 20)
>>> output = m(input)
>>> print(output.size())
torch.Size([128, 30])
```

## `torch.squeeze(input, dim=None, out=None) → Tensor`
返回一个张量，其中输入张量的所有维度中大小为1的将会被剔除。
范例，$(A*1*B*C*1*D)$对应的输出为$(A*B*C*D)$
注意，`dim=None`该参数一旦给定，意味着在给定的维度上进行剔除工作。

## `torch.div(input, other, out=None) → Tensor`
将输入输入的每个元素除以标量其他，然后返回一个新的结果张量。

<center>

<font size =5> 

$out_i=\frac{input_i}{other}$

</font>

</center>

```python
>>> a = torch.randn(4, 4)
>>> a
tensor([[-0.3711, -1.9353, -0.4605, -0.2917],
        [ 0.1815, -1.0111,  0.9805, -1.5923],
        [ 0.1062,  1.4581,  0.7759, -1.2344],
        [-0.1830, -0.0313,  1.1908, -1.4757]])
>>> b = torch.randn(4)
>>> b
tensor([ 0.8032,  0.2930, -0.8113, -0.2308])
>>> torch.div(a, b)
tensor([[-0.4620, -6.6051,  0.5676,  1.2637],
        [ 0.2260, -3.4507, -1.2086,  6.8988],
        [ 0.1322,  4.9764, -0.9564,  5.3480],
        [-0.2278, -0.1068, -1.4678,  6.3936]])
```