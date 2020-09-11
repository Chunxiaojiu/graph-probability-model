# 理解LSTM 网络

## RNN

人类不会每次都从头开始思考，当你阅读这篇文章的时候，我们可以理解每一个文字通过前一个文字的理解。你并没有从头开始思考放弃所有之前的认知，你的认知是持续的。传统的神经网络无法做到这一点，这似乎是一个重大缺陷。例如，假设您想对电影中每个点发生的事件进行分类。尚不清楚传统的神经网络如何利用其对电影中先前事件的推理来告知后期事件。递归神经网络解决了这个问题。它们是具有循环的网络，可以使信息持久存在。
<br>
<br>
<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-rolled.png" width = "150" height = "200"/>
<br>
<div style = "color:#999">递归神经网络有循环</div>
</center>

在上图中是一大块的神经网络，$A$ 是通过输入的 $x_{t}$ 和 $h_{t}$ 来计算。循环允许信息从网络的一个步骤传递到下一个步骤。这些循环使得递归神经网络显得比较神秘。然而，我们稍加思索就会发现递归神经网络和原始的神经网络并非完全不同。一个递归神经网络可以思考成多个相同网络的拷贝，同时每一个传递信息给他们的后继成员。接下来让我们考虑一下在循环里发生了什么，我们拆开这个结构体可以看到下面：


<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-unrolled.png" width = 100% height = "200"/>
<br>
<div style = "color:#999">递归神经网络展开循环</div>
</center>

## simple RNN和 BPTT算法推导
未完成

----
----
这种类似链的性质表明，递归神经网络与序列和列表密切相关。它们是用于此类数据的神经网络的自然架构。而且它们肯定被使用了！在过去的几年中，将RNN应用到各种问题上已经取得了令人难以置信的成功：语音识别，语言建模，翻译，图像字幕…清单还在继续。我将在Andrej Karpathy的出色博客文章 ["递归神经网络的不合理有效性"](http://karpathy.github.io/2015/05/21/rnn-effectiveness/) 中讨论使用RNN可以实现的惊人功能,但是他们真的很棒。

这些成功的关键是使用“LSTM”，这是一种非常特殊的递归神经网络，它在许多任务上都比标准版本好得多。利用递归神经网络几乎可以实现所有令人兴奋的结果。本文将探讨的是这些LSTM。

## 长期依赖问题
**RNN**一个重要的作用是可以将过去的信息连接到现在的任务上，就像我们可以使用之前的视频帧画面来对现在的帧画面进行理解。如果**RNN**能够完全做到这一点那这种方式确实是非常的有用的。但是他们真的可以做到么？

有时，我们只需要查看最新信息即可执行当前任务。例如，考虑一种语言模型，该模型试图根据前一个单词预测下一个单词。如果我们试图预测“云在天空中”的最后一个词，那么我们不需要任何进一步的上下文————很明显，下一个词将是天空。在这种情况下，相关信息与所需信息之间的差距很小，RNN可以学习使用过去的信息。

<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-shorttermdepdencies.png">
<div style = "color:#999">短期的递归神经网络</div>
</center>


但是在某些情况下，我们需要更多背景信息。考虑尝试预测文本“我在法国长大……我会说流利的 *法语* ”中的最后一个词。最近的信息表明，下一个词可能是一种语言的名称，但是如果我们想缩小哪种语言的范围，我们需要从更远的地方来追溯法国的情况。相关信息和需要扩大的点之间的差距完全可能。但是随着差距的扩大，RNN没办法学习去关联这些信息。

<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/RNN-longtermdependencies.png">
<div style = "color:#999">差距较大时的递归神经网络，可以发现无法将最开始的信息和后续信息关联</div>
</center>



从理论上来说，RNN是完全可以处理这种“长期依赖”。一个人类可以通过谨慎的挑选参数来解决这种形式的简单问题。但是很遗憾，在实际过程中**RNN**看起来并不能很好的处理这种问题。该问题在 Hochreiter (1991)  [German] [^1]
和Bengio, et al. (1994)[^2]中进行了深入的讨论并说明了为什么做到这很困难。



## LSTM 网络

长短期记忆网络————通常叫做“LSTMs”，也是一种特殊的RNN，可以处理学习长期依赖的问题。该网络由Hochreiter & Schmidhuber (1997)[^3]定义并在后续的工作中受到很多人的使用和欢迎。LSTM被明确设计为避免长期依赖问题。长时间记住信息实际上是他们的默认行为，而不是他们努力学习的东西！
<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-SimpleRNN.png">
<div style = "color:#999">传统的RNN</div>
</center>

所有的递归神经网络都具有神经网络的重复模块链的形式。在标准RNN中，此重复模块将具有非常简单的结构，例如单个tanh层。 而LSTM同样也具有这种链式结构，但是重复的这个模型有着不一样的结构。与其不同的LSTM拥有4个神经网络层，并以一种特殊的方式进行交互。

<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png">
<br>
<div style = "color:#999">LSTM</div>
</center>


不用担心现在我们还无法看懂这个图，我们接下来将一步步解析，首先我们先明确一下这里面的标记：

<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM2-notation.png" width = 75% >
<br>
<div style = "color:#999">LSTM的变量声明</div>
</center>

在上图中，每条线都承载着整个矢量，从一个节点的输出到另一节点的输入。粉色圆圈表示逐点操作，如矢量加法，而黄色框表示学习的神经网络层。合并的行表示串联，而分叉的行表示要复制的内容，并且副本到达不同的位置。


## LTSM的核心思想
LSTM的关键是单元状态，水平线贯穿图的顶部。单元状态有点像传送带。 它沿整个链条一直沿直线延伸，只有一些较小的线性相互作用。 信息不加改变地流动非常容易。LSTM确实具有删除或向单元状态添加信息的能力，这些功能由称为<font size = 5>**门**</font>的结构精心调节。门是一种具有信息选择的通路，它们是由sigmod神经网络层和逐点乘法运算构成的。
<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-C-line.png" width = 75% >
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-gate.png" width = 15% >
</center>


$sigmod$ 层输出的数值在0-1之间，它描述了对于每一个组件它是否可以通过的可能。0对应的是 **“不让任何一个通过”**，1对应的是 **“让所有都通过”**。一个LSTM网络拥有三种门来控制和保护每一个单元的状态。



## LSTM分步介绍

第一步我们需要决定在LSTM单元状态中我们需要抛弃哪些信息。这个决定是由sigmod层中的 **“遗忘门”** 来进行决定的。制作这个决定的sigmod层被称为 **遗忘门层** ，它通过获取 $h_{t-1}$ 和 $x_{t}$ 来对于单元状态 $C_{t-1}$ 中的每一个数字输出一个0-1之间的数。**1**代表了“完整的保留下这个数”，**0**代表了”完全的放弃这个数“。
让我们回到语言模型的示例，该模型试图根据所有先前的单词来预测下一个单词。 在这样的问题中，单元状态可能包括当前受试者的性别，从而可以使用正确的代词。 看到新主题时，我们想忘记旧主题的性别。
<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png" width = 75% >
<br>
</center>


下一步是确定我们将在单元状态下存储哪些新信息。 这包括两个部分。 首先，称为“输入门层”的 $sigmod$ 层决定了我们将更新哪些值。然后，$tanh$ 层创建一个新候选值的向量，$\widetilde{C_{t}}$,然后将它加入到这个状态中。在下一个步骤中，我们将把这两个结合去创造一次状态的更新。
<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png" width = 75% >
<br>
</center>


在我们的语言模型示例中，我们想将新主题的性别添加到单元格状态，以替换我们忘记的旧主题。现在是时候将过去状态$C_{t-1}$更新成$C_{t}$。过去的步骤已经确定了要做什么，我们接下来只需要继续进行即可。

我们将旧的状态乘以$f_{t}$,来忘记我们之前决定要忘记的事情。然后我们加上$i_{t}*\widetilde{C_{t}}$ 。这是根据我们决定更新每个状态值大小进行缩放后的新的候选值。
就语言模型而言，这是我们实际删除旧主题的性别的信息并添加新信息的地方，正如我们在前面的步骤中所确定的那样。
<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png" width = 75% >
<br>
</center>


最后我们需要决定我们到底需要输出什么。此输出将基于我们的单元状态，但将是过滤后的版本。 首先，我们运行一个$sigmod$层，确定要输出的单元状态的哪些部分。接下来我们将单元状态通过$tanh$(将输入值转变成-1~1)同时将$sigmod$门的输出乘以该值，通过这些我们仅仅输出我们决定的部分。


对于语言模型示例，由于它只是看到一个主语，因此可能要输出与动词相关的信息，以防万一。 例如，它可能会输出主语是单数还是复数，以便我们知道如果接下来要动词的动词应变形为哪种形式。
<center>
<img src = "http://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png" width = 75% >
<br>
</center>


[^1]:http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf
[^2]:http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf
[^3]:http://www.bioinf.jku.at/publications/older/2604.pdf



