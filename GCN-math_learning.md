## 图卷积网络（GCN）
我们知道feature的传播可以看作一个热传导模型，我们通过贝叶斯网络的介绍可以知道，在贝叶斯无向图中我们是采取一个势函数来对结点进行更新的，所以我们先从一维的温度传导开始：
<img src = https://pic1.zhimg.com/80/v2-f67aa2769d22eb54650a1077bb921fcf_720w.jpg >
可以发现对于第$i$个单元而言，影响它温度变化的外部原因是它周围的两个点，也就是$i-1$和$i+1$两个单元，然后我们对于每个结点的温度设为$\phi_i$，那么温度传导有如下形式：
$$
\frac{d\phi_i}{dt} = k(\phi_{i+1}-\phi_i) - k(\phi_i - \phi_{i-1})\\
||\\
\frac{d\phi_i}{dt} - k[(\phi_{i+1}-\phi_i) - (\phi_i - \phi_{i-1})] = 0
$$

注意到这里出现了一个两者差分的差分，这是在离散化空间下的表示，如果是在连续空间上，我们会轻松的发现，这其实就是一个二阶导数。
所以我们可以简单的将热传导方程写出来：
$$\frac{\partial\phi}{\partial t} -k\frac{\partial^2\phi}{\partial x^2} = 0$$
$$\frac{\partial\phi}{\partial t} - k\Delta\phi = 0$$

其中$\Delta$就是我们熟悉的拉普拉斯算子。