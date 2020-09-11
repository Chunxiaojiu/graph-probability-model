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

其中$\Delta$就是我们熟悉的拉普拉斯算子。接下来我们从一维图开始走向多维度,现在，我们依然考虑热传导模型，只是这个事情不发生在欧氏空间了，发生在一个Graph上面。这个图上的每个结点（Node）是一个单元，且这个单元只和与这个结点相连的单元，也就是有边（Edge）连接的单元发生热交换。例如下图中，结点1只和结点0、2、4发生热交换，更远的例如结点5的热量要通过4间接的传播过来而没有直接热交换。

<img src = https://pic4.zhimg.com/80/v2-95364a51f5601bd3981458fd805942ee_720w.jpg>

根据上面的图结合牛顿冷却定律，我们可以将每个点温度随着时间变化写成如下：
$$
\frac{d\phi_i}{dt} = -k\sum _j A_{ij}(\phi_i-\phi_j)
$$

其中$A_{ij}$是这个图的邻接矩阵，然后这个图是一个无向无环图，我们将上式进行推导：
$$
\frac{d\phi_i}{dt} = -k[\phi_i\sum _j A_{ij} - \sum _j A_{ij}\phi_j]=-k[deg(i)\phi_i- \sum_j A_{ij}\phi_j]
$$
先看右边括号里面第一项， $deg(·)$代表对这个顶点求度（degree），一个顶点的度被定义为这个顶点有多少条边连接出去，然后我们定义向量$\phi = [\phi_1,\phi_2...,\phi_n]^T$,$D = diag(deg(1),deg(2)...deg(n))$，D也被称为度矩阵，只有对角线上有值。整理上式可得：
$$
\frac{d\phi_i}{dt}  = -kD\phi +kA\phi = -k(D-A)\phi
$$
然而在欧式空间下的微分方程为：
$$
\frac{\partial\phi}{\partial t} - k\Delta\phi = 0
$$
那D-A和拉普拉斯算子有着同样的形式，只不过一个在离散空间上，一个在欧式空间下的连续，于是拉普拉斯矩阵就出现了：
<img src =https://pic3.zhimg.com/80/v2-5f9cf5fdeed19b63e1079ed2b87617b4_720w.jpg >

同样拉普拉斯矩阵也有不同的种类，常用的有以下三种：
- $L=D-A$ Combinatorial laplacian
- $L^sys = D^{-1/2} LD^{-1/2} $ Symmetric normalized Laplacian
- $L^{rw} = D^{-1}L$ Random walk normalized Laplacian
  
到此我们了解了为什么我们需要使用拉普拉斯矩阵，但是我们现在是要进行图卷积，那卷积来提取特征又是如何实现的呢？这就不得不提到特征向量和拉普拉斯举证的谱分解（特征分解）。
假设我们在这里使用标准化后的拉普拉斯矩阵进行卷积，也就是我们上面提到的$Symmetric \quad normalized\quad Laplacian$ 那么$L = I- D^{-1/2}WD^{-1/2}$,而对应的邻接矩阵$A = D^{-1/2}WD^{-1/2}$,$v_i$表示每个点的温度或者说是势能:
$$
\begin{cases}
     A_{ii} = 0  \quad 无自环\\
     A_{ij} = 1/d \quad ,i,j有边\\
     A_{ij} = 0 \quad ,i,j没有边
\end{cases}\\
(Av)_i = \frac{1}{d}\sum _{j: (i,j)\in E }  v_j
$$
拉普拉斯矩阵，不同于邻接矩阵在描述运动的轨迹，**laplace矩阵描述的是运动的位移或者变化量**。
$$
(Av)_i = \frac{1}{d}\sum_{j: (i,j)\in E }  v_j\\
(Lv)_i = v_i - \frac{1}{d}\sum_{j: (i,j)\in E }  v_j
$$
显然这个变化量可以经过多次的迭代：
$$
Lv = \mu_1\alpha_1u_1+\mu_2\alpha_2u_2+... +\mu_n\alpha_nu_n \\
Lv^2 = \mu_1^2\alpha_1u_1+\mu_2^2\alpha_2u_2+... +\mu_n^2\alpha_nu_n \\ 
Lv^k = \mu_1^k\alpha_1u_1+\mu_2^k\alpha_2u_2+... +\mu_n^k\alpha_nu_n
$$
于是，L的特征值描述了变化量的强度。事实上，这只是特征值的冰山一角，特征值描述的信息比你想象中更多，接来下，我们介绍一下，特征值与图的的另一种联系
### Variational Characterization of Eigenvalues（特征值变异表征）
