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
- $L^sys = D^{-1/2}LD^{-1/2}$ Symmetric normalized Laplacian
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
定理如下：设$M\in R^{n*n}$是一个对称矩阵，，同时M的特征值分别为$\alpha_1\leq\alpha_2\leq....\leq\alpha_n$,于是
$$
\alpha_k = \min _{k-dimV} \max _{x\in V- \{ 0\}} \frac{x^TMx}{x^Tx}$$

其中$\frac{x^TMx}{x^Tx}$被称为x在M中的瑞利商（Rayleigh Quotient），并记为$R_M(x)$。
这个地方的证明我们可以通过证明瑞利定理的同样方式来证明[1],[2]

[1]:https://blog.csdn.net/klcola/article/details/104800804
[2]:https://www.cnblogs.com/xingshansi/p/6702188.html?utm_source=itdadao&utm_medium=referral

这个定理非常神奇且重要，他告诉我们，当x的取值在一个由K个特征向量张成的子空间中，$\frac{x^TMx}{x^Tx}$的最大值恰好等于特征值。那这会让我们联想到PCA主成分分析法中的特征值问题，那我们试着分析一下这里的特征值又有什么作用，和我们的GCN的联系在哪呢？

那我们接下来从**谱聚类**来开展，谱聚类(spectral clustering)是一种针对图结构的聚类方法，它跟其他聚类算法的区别在于，他将每个点都看作是一个图结构上的点，所以，判断两个点是否属于同一类的依据就是，两个点在图结构上是否有边相连，可以是直接相连也可以是间接相连。

<img src = https://img-blog.csdnimg.cn/20190331150947630.png>

那我们一般使用图模型进行分类判别的时候，我们经常会使用到如下的一个目标函数：
$$x^TMx = \sum_{\{u,v\}\in E} (x_u-x_v)^2$$

<center>
<img src = https://img-blog.csdnimg.cn/20190331150932988.png>
</center>

那我们在使用上面提到的目标函数来解决上图分类问题的时候相当于在做一个$x\in{\{0,1}\}^V$的问题
$$\min \sum_{\{u,v\}\in E} (x_u-x_v)^2 = \sum_{u\in A,v\notin A} (x_u-x_v)^2 = 2cut(A,\overline{A})$$

然后,我们将其转化成拉普拉斯矩阵$M = D-A$ 可以清楚的看到：
$$x^T(M)x = x^T(dI-A)x =dx^Tx-x^TAx = \sum_v dx_v^2 - 2\sum_{\{u,v\}\in E}x_ux_v =\sum_{\{u,v\}\in E} (x_u-x_v)^2 $$
经过上面的简单操作之后我们发现我们将拉普拉斯矩阵和目标函数很好的结合到了一起，如果把上面的拉普拉斯矩阵的转移矩阵中的数值带上权重$w_{ij}$，那我们可以得到下面的推广式子：
$$\sum_{ij} w_{ij}(x_u - x_v)^2$$

那接下来我们考虑一个图像多簇分割问题，然后将我们上面计算出来的结果带入到这个问题中进行求解
问题中假设我们有k个簇对应的要最小化$cut(A_1,A_2,A_3...A_k)$同时我们还要最大化每个子分割的数目，其中：
$$cut(A_i,\overline{A_i}) = \sum_{i\in A,j\notin A} w_{ij}$$
同时我们引入指示向量$h_j \in \{ h_1,h_2...,h_k\}$,对于任意的$h_j$向量，他是一个|V|-维度的向量（|V|为该子图的结点数），定义如下：
$$
h_{ij} = 
\begin{cases}
0\quad v_j \notin A_i\\
\frac{1}{\sqrt{|A_i|}} \quad \in A_i
\end{cases}
$$

那对与每个子图我们可以得到如下计算：
$$\begin{aligned}
h_{i}^{T} L h_{i} &=\frac{1}{2} \sum_{m=1}^{|V|} \sum_{n=1}^{|V|} w_{m n}\left(h_{i m}-h_{i n}\right)^{2} \\
&=\frac{1}{2}\left[\sum_{m\in A_i,n\notin A_i} w_{mn}(\frac{1}{\sqrt{|A_i|}} -0)^2 +\sum_{m\notin A_i,n\in A_i} w_{mn}(0 - \frac{1}{\sqrt{|A_i|}})^2 \right] \\
&=\frac{1}{2}\left(\sum_{m \in A, n \notin A_{i}} w_{m n} \frac{1}{\left|A_{i}\right|}+\sum_{m \notin A_{i}, n \in A_{i}} w_{m n} \frac{1}{\left|A_{i}\right|}\right)\\
&=\frac{1}{2}\left(\operatorname{cut}\left(A_{i}, \bar{A}_{i}\right) \frac{1}{\left|A_{i}\right|}+\operatorname{cut}\left(\bar{A}_{i}, A_{i}\right) \frac{1}{\left|A_{i}\right|}\right) \\
&=\frac{\operatorname{cut}\left(A_{i}, \bar{A}_{i}\right)}{\left|A_{i}\right|}
\end{aligned}$$

上面的推导原理很简单，如果两个点都处在或都不处于$A_i$时都为0，其他为$\frac{1}{\sqrt{|A_i|}}$。如果我们将K个子图h合并成为一个大的矩阵H，上述式子变成如下：
$$ Cut(A_1,A_2,A_3...A_k) = \sum ^k _{i = 1} h_i ^T L h_i = \sum ^k _{i = 1}(H^TLH)_{ii} = tr(H^TLH)\\ s.t. \quad h_i^Th_i = 1$$

推导出上面式子之后，我们的目标时最小化$tr(H^TLH)$,很容易想到最优化的方案，也就是采用拉普拉斯不等式将限制条件加入等式中进行求解，注意到$h_i$是相互正交的（因为每个点只能出现在其中一个子集中，没有交叠）
$$\begin{aligned}
\nabla_{h}\left(h^{T} L h-\lambda\left(1-h^{T} h\right)\right) &=\nabla_{h} t r\left(h^{T} L h-\lambda\left(1-h^{T} h\right)\right) \\
&=\nabla_{h} t r\left(h^{T} L h\right)-\lambda \nabla_{h} t r\left(h h^{T}\right) \\
&=\nabla_{h} t r\left(h h^{T} L\right)-\lambda \nabla_{h} t r\left(h E h^{T} E\right) \\
&=\nabla_{h} \operatorname{tr}\left(h E h^{T} L\right)-\lambda \nabla_{h} t r\left(h E h^{T} E\right) \\
&=L h+L^{T} h-\lambda(h+h) \\
&=2 L h-2 \lambda h \\
&=0 \\
& \Longrightarrow L u=\lambda h
\end{aligned}$$

其实这个推导跟PCA是一样的，只不过PCA找的是最大特征值（PCA中L是协方差矩阵，目标是找到一个向量最大化方差），这里是找最小特征值，我们目标是找到一个向量最小化这个二次型矩阵。那回到我们最开始讲到的$\frac{x^TMx}{x^Tx}$的最大值恰好等于特征值，那M等于标准化的拉普拉斯矩阵L时，$x\in\{0,1\}^V$那么该公式就变成了表示分开不同类别的边数：
$$\mathbf{x}^{T} L \mathbf{x}=\frac{1}{d} \sum_{\{u, v\} \in E}\left(x_{u}-x_{v}\right)^{2}$$
$$\lambda_{k}=\min _{S k-\text { dimensional subspace of } \mathbb{R}^{n}} \max _{\mathbf{x} \in S-\{0\}} \frac{\sum_{\{u, v\} \in E}\left(x_{u}-x_{v}\right)^{2}}{d \sum_{v} x_{v}^{2}}$$

那此时$\lambda_k = 0$显然代表着在图中找到$A_k$集合，满足所有点都属于同一个类，形成一个完美的孤岛，因此，**laplace矩阵特征值为0的个数就是连通区域的个数**。此外，在真实场景中可能不存在特征值为0的孤岛（因为每个人或多或少都会跟人有联系），但是当特征值很小的时候，也能反应出某种“孤岛”，或者称为，bottleneck，这种bottleneck显然是由第二小的特征值决定的（因为特征值为0就是真正孤岛，但第二小就是有一点点边，但还是连通的），因此，很多发现社交社群的算法都会或多或少利用利用这一点，因为不同的社群肯定是内部大量边，但是社群之间的边很少，从而形成bottlenect。
定理: 设G为无向图, A为G的邻接矩阵, 令$L=I-\frac{1}{d} A$ 为图G的标准化的laplace矩阵。设 $\lambda_{1} \leqslant \lambda_{2} \leqslant \ldots \lambda_{n},$ 为L的特征值, 大小按照递增排列，则:
1. $\lambda_{1}=0$ 且 $\lambda_{n} \leqslant 2$
2. $\lambda_{k}=0$ 当目仅当G至少k个连通区域（这意味着特征值为0的数量对应若连通区域的数量）。
3. $\lambda_{n}=2$ 当目仅当至少有一个连通域是二分图。 

到此我们将谱分解使用在了聚类算法上，但是我们的最终目标是实现图卷积网络，卷积要如何实现呢？
？为了解决这个问题题，我们可以利用图上的傅里叶变换，再使用卷积定理，这样就可以通过两个傅里叶变换的乘积来表示这个卷积的操作。那么为了介绍图上的傅里叶变换，我接来下从最原始的傅里叶级数开始讲起。
<img src = https://img-blog.csdnimg.cn/20190330233251671.gif>

在将图的傅里叶变换之前，我们先介绍一下图信号是什么。我们在传统概率图中，考虑每个图上的结点都是一个*feature*，对应数据的每一列，但是图信号不一样，这里每个结点不是随机变量，相反它是一个object。也就是说，他描绘概率图下每个样本之间的图联系，可以理解为刻画了不满足$i.i.d$假设的一般情形。
<img src = https://img-blog.csdnimg.cn/20190330234154754.png>

在这里我们将每一个信息输入的点（*feature*）记为$f(i)$,$\mu_l(i)$记录为第l个特征向量的第i个分量。对应将其中一个输入用正交基打开如下：
$$f(i)=\lambda_{1} u_{1}(i)+\cdots+\lambda_{l} u_{l}(i)+\cdots+\lambda_{N} u_{N}(i)$$

而拉普拉斯矩阵根据之前的特征值分解，我们可以拆解成下面的形式：
$$L=U\left(\begin{array}{ccc}
\lambda_{1} & & \\
& \ddots & \\
& & \lambda_{n}
\end{array}\right) U^{-1}$$

在卷积网络中我们需要构建一个卷积核进行更新迭代，假定一个卷积核h与特征值相关联，用于迭代更新，则我们定义的网络卷积将变成如下形式：
$$(f * h)_{G}=U\left(\begin{array}{ccc}
\hat{h}\left(\lambda_{1}\right) & & \\
& \ddots & \\
& & \hat{h}\left(\lambda_{n}\right)
\end{array}\right) U^{T} f = (f * h)_{G}=U\left(\left(U^{T} h\right) \odot\left(U^{T} f\right)\right)$$
$\odot$表示Hadamard product（哈达马积），对于两个维度相同的向量矩阵张量进行对应位置的逐元素乘积运算。