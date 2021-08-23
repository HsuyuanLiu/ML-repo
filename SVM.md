# SVM

支持向量机（support vector machines，SVM）是一种二分类模型。**它的基本模型是定义在特征空间上的间隔最大的线性分类器**，这也是其与普通感知机的区别所在；其学习策略就是间隔最大化，可形式化为一个求解凸二次规划的问题，也等价于正则化的合页损失函数的最小化问题。

当输入空间为欧氏空间或离散集合、特征空间为希尔伯特空间时，核函数表示将输入从输入空间映射到特征空间得到的特征向量之间的内积。通过使用核函数可以学习非线性支持向量机，等价于隐式地在高维的特征空间中学习线性支持向量机。

### 线性可分

线性可分的定义可以参见感知器，而对于优化问题，首先需要设置一个损失函数：

##### 函数间隔与几何间隔

**一般来说，一个点距离分离超平面的远近可以表示分类预测的确信程度。**在超平面$w·x+b$＝0确定的情况下，$|w·x+b|$能够相对地表示点x距离超平面的远近。而$w·x+b$的符号与类标记$y$的符号是否一致能够表示分类是否正确。所以可用量$y(w·x+b)$来表示分类的正确性及确信度，这就是函数间隔的概念：
$$
\hat \gamma=y_i(w \cdot x_i+b)
$$
即可以定义出该超平面关于训练数据集的函数间隔为超平面关于该数据集$T$中所有样本点的函数间隔的最小值，即
$$
\hat \gamma= \min_{i=1,\cdots,N}\hat \gamma_i
$$
但是选择分离超平面时，单一的函数间隔还不够。因为只要成比例地改变$w$和$b$，例如将它们改为$2w$和$2b$，超平面并没有改变，但函数间隔却成为原来的2倍。这一事实启示我们，可以对分离超平面的法向量$w$加某些约束，如规范化，规定$||w||＝ 1$，使得间隔是确定的。这时函数间隔成为几何间隔：
$$
\hat \gamma=\frac{w}{||w||} \cdot x_i+\frac{b}{||w||}
$$
其中，$||w||$为$w$的$L_2$范数。

此时模仿上文，引入负样本的计算，即可以得到最终的距离计算公式：
$$
\hat \gamma=y_i(\frac{w}{||w||} \cdot x_i+\frac{b}{||w||})
$$
在此基础上，可以定义该平面到到数据集的距离即为$\min \hat \gamma$，即最近的点到该平面的距离.

##### 间隔最大化

对线性可分的训练数据集而言，线性可分分离超平面有无穷多个（等价于感知机），但是几何间隔最大的分离超平面是唯一的，这里的间隔最大化又称为硬间隔最大化。其直观解释是：对训练数据集找到几何间隔最大的超平面意味着以充分大的确信度对训练数据进行分类。也就是说，不仅将正负实例点分开，而且对最难分的实例点（离超平面最近的点）也有足够大的确信度将它们分开。这样的超平面应该对未知的新实例有很好的分类预测能力。

这一问题可以表述为以下优化问题：
$$
\begin{gathered}
\max_{w,b} \qquad \gamma \\
s.t.\qquad y_i(\frac{w}{||w||} \cdot x_i+\frac{b}{||w||}) \geq \hat \gamma, \qquad i=i,2...N
\end{gathered}
$$
此时不妨令$\gamma=1$，且最大化$\frac{1}{||w||}$与最小化$\frac{1}{2}{||w||}^2$相互等价。，因此该优化问题可以进一步改写为：
$$
\begin{gathered}
\max_{w,b} \qquad \frac{1}{2}{||w||}^2 \\
s.t.\qquad y_i(w \cdot x_i+b)-1 \geq 0 \qquad i=i,2...N
\end{gathered}
$$
这样就转化为了一个凸优化问题的形式：
$$
\begin{gather}
\min_w \qquad f(w) \\
s.t \qquad g_i(w) \le 0 \qquad i=1,2,...k\\
h_i(w)=0 \qquad i=1,2,...k
\end{gather}
$$
目标函数$f(w)$和约束函数$g_i(w$)都是$R^n$上的连续可微的凸函数，约束函数$h_i(w)$是$R^n$上的仿射函数,且当目标函数$f(w$)是二次函数且约束函数$g_i(w$)是仿射函数时，上述凸最优化问题成为凸二次规划问题。

**可以证明，最大间隔分离超平面一定存在且唯一。**

**在线性可分情况下，训练数据集的样本点中与分离超平面距离最近的样本点的实例称为支持向量（support vector）。支持向量是使约束条件式取等的数据点。**

在决定分离超平面时只有支持向量起作用，而其他实例点并不起作用。如果移动支持向量将改变所求的解；但是如果在间隔边界以外移动其他实例点，解是不会改变的。由于支持向量在确定分离超平面中起着决定性作用，所以将这种分类模型称为支持向量机。支持向量的个数一般很少，所以支持向量机由很少的“重要的”训练样本确定。

### 求解算法

为了求解线性可分支持向量机的最优化问题, 将其作为原始最优化问题，应用拉格朗日对偶性）, 通过求解`对偶问题`得到原始问题的最优解, 这就是线性可分支持向量机的对偶算法。这样做的优点, 一是对偶问题往往更容易求解; 二是自然引入核函数, 进而推广到非线性分类问题。 其步骤如下：

**首先构建拉格朗日函数**。为此, 对每一个不等式约束引进拉格朗日乘子$\mathrm{a} \geq 0, \mathrm{i}=1,2, \ldots, \mathrm{N}$,

**定义拉格朗日函数:**
$$
L(w, b, \alpha)=\frac{1}{2}\|w\|^{2}-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(w \cdot x_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i}
$$
其中, $\mathrm{a}=\left(\mathrm{a}_{1}, \mathrm{a}_{2}, \ldots, \mathrm{a}_{\mathrm{N}}\right)^{T}$为拉格朗日乘子向量。
根据拉格朗日对偶性, 原始问题的对偶问题是极大极小问题：
$$
\max _{w, b} \min _{w, b} L(w, b, \alpha)
$$
所以, 为了得到对偶问题的解，需要先求 $\mathrm{L}(\mathrm{w}, \mathrm{b}, \mathrm{a})$ 对 $\mathrm{w}, \mathrm{b}$ 的极小, 再求对a的极大。

- 求 $\min _{w, b} L(w, b, \alpha)$
  将拉格朗日函数 $L(w, b, a)$ 分别对 $w, b$ 求偏导数并令其等于0。

$$
\begin{aligned}
&\nabla_{w} L(w, b, \alpha)=w-\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}=0 \\
&\nabla_{b} L(w, b, \alpha)=\sum_{i=1}^{N} \alpha_{i} y_{i}=0
\end{aligned}
$$
得
$$
w=\sum_{i=1}^{N} \alpha_{i} y_{i} x_{i}
$$
$$
\sum_{i=1}^{N} \alpha_{i} y_{i}=0
$$

代入原式，即可知：
$$
\begin{aligned}
L(w, b, \alpha) &=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} y_{i}\left(\left(\sum_{j=1}^{N} \alpha_{j} y_{j} x_{j}\right) \cdot x_{i}+b\right)+\sum_{i=1}^{N} \alpha_{i} \\
&=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
\end{aligned}
$$
即原始优化函数可以化简为：
$$
\min _{x, b} L(w, b, \alpha)=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
$$
此时若求 $\min _{w, b} L(w, b, \alpha)$对 } a的极大, 即是求对偶问题的解：
$$
\begin{array}{ll}
\max_{\alpha} & -\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i} \\
\text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
& \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N
\end{array}
$$
将上式目标函数由求极大转换成求极小, 就得到下面与之等价的对偶最优化问题:
$$
\begin{array}{ll}
\min _{\alpha} & \frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)-\sum_{i=1}^{N} \alpha_{i} \\
\text { s.t. } & \sum_{i=1}^{N} \alpha_{i} y_{i}=0 \\
& \alpha_{i} \geqslant 0, \quad i=1,2, \cdots, N
\end{array}
$$
在此基础上即可以通过`SMO`等方法接触$\alpha$的具体值，并代回原式，写出最终结果：
$$
w^*=\sum_{i=1}^{N} \alpha_{i}^* y_{i} x_{i}
$$

$$
b^*=y_j-\sum_{i=1}^{N} \alpha_{i}^* y_{i} (x_{i}\cdot x_j)
$$

即可以写出最终的结果：
$$
w^* \cdot x+b^*=0
$$
在这一情况下，即得到了在线性可分情况下的最优解。

### 软间隔最大化

对线性不可分训练数据，上述方法中的不等式约束并不能都成立。因此需要软间隔算法对其进行处理。

线性不可分意味着某些样本点$(x_i，y_i)$不能满足函数间隔大于等于1的约束条件。为了解决这个问题，可以对每个样本点$(x_i，y_i)$引进一个松弛变量$i≥0$，使函数间隔加上松弛变量大于等于1。这样，约束条件变为：
$$
y_i(w \cdot x_i+b)>1-\xi
$$
同时对于每一个松弛变量，支付某一个代价$C\xi$，因此，最终的代价函数可以写为：
$$
\frac{1}{2}\|w\|^{2}+C \sum_i^N \xi
$$
这里，$C>0$称为惩罚参数。

因此线性不可分的线性支持向量机的学习问题变成如下凸二次规划问题：
$$
\begin{array}{ll}
\min _{w, b, \xi} & \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i} \\
\text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N \\
& \xi_{i} \geqslant 0, \quad i=1,2, \cdots, N
\end{array}
$$
该问题关于 $(\mathrm{w}, \mathrm{b}, \boldsymbol{\xi})$ 的解是存在的，可以证明w的解是唯一的, 但b的解不唯一而应当是一个区间。

其解决方法与之前的算法基本相同。

### 另一种理解

除了以上的计算过程外，线性支持向量机学习还有另外一种解释, 就是最小化以下目标函数:
$$
\sum_{i=1}^{N}\left[1-y_{i}\left(w \cdot x_{i}+b\right)\right]_{+}+\lambda\|w\|^{2}
$$
目标函数的第一项是经验损失或经验风险函数：
$$
L(y(w \cdot x+b))=[1-y(w \cdot x+b)]_{+}
$$
称为合页损失函数（hinge loss function）。下标“ $+$ "表示以下取正值的函数。
$$
[z]_{+}= \begin{cases}z, & z>0 \\ 0, & z \leqslant 0\end{cases}
$$
在这一理解下，原式可以被看做为神经网络中对经验损失函数与$L_2$正则化项的加和，即其目标即在于最小化损失函数与惩罚项之和。

基于此，原始的优化问题：
$$
\begin{array}{ll}
\min _{w, b, \xi} & \frac{1}{2}\|w\|^{2}+C \sum_{i=1}^{N} \xi_{i} \\
\text { s.t. } & y_{i}\left(w \cdot x_{i}+b\right) \geqslant 1-\xi_{i}, \quad i=1,2, \cdots, N \\
& \xi_{i} \geqslant 0, \quad i=1,2, \cdots, N
\end{array}
$$
等价于最优化问题
$$
\min _{w, b} \sum_{i=1}^{N}\left[1-y_{i}\left(w \cdot x_{i}+b\right)\right]_{+}+\lambda\|w\|^{2}
$$
在$\lambda=\frac{1}{2C}$时是成立的；

### 核技巧

非线性分类问题是指通过利用非线性模型才能很好地进行分类的问题，在这一情况下，希望通过一个非线性变换将输入空间（欧氏空间$\mathbb{R^n}$或离散集合）对应于一个特征空间（希尔伯特空间）$\mathcal{H}$，使得在输入空间$\mathbb{R^n}$中的超曲面模型对应于特征空间中的超平面模型（支持向量机）。这样，分类问题的学习任务通过在特征空间中求解线性支持向量机就可以完成。

##### 核函数

设$x$是输入空间（欧氏空间$\mathbb{R^n}$的子集或离散集合），又设$\mathcal{H}$为特征空间（希尔伯特空间），若存在一个从$x$到$\mathcal{H}$的映射，满足：
$$
\Phi(x):\mathcal{X} \rightarrow \mathcal{H}
$$
使得对所有$x,z\in \mathcal{X}$，函数$K(x,z)$满足条件
$$
K(x,z)=\phi(x)\cdot \phi(z)
$$
则称$K(x,z)$为核函数，$\phi(x)$为映射函数，式中$\phi(x)\cdot \phi(z)$为$\phi(x)$和$\phi(z)$的内积。

在实际应用中，一般的做法是，在学习与预测中只定义核函数K(x,z)，而不显式地定义映射函数$\phi(x)$。

##### 来源

核技巧的来源在于对以下公式的某种改良：
$$
\min _{x, b} L(w, b, \alpha)=-\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}\left(x_{i} \cdot x_{j}\right)+\sum_{i=1}^{N} \alpha_{i}
$$
在这一过程中，将$(x_1\cdot x_2)$替换为$K(x_i,x_j)$，及可以得到带有核函数的计算公式：
$$
W(\alpha)=\frac{1}{2} \sum_{i=1}^{N} \sum_{j=1}^{N} \alpha_{i} \alpha_{j} y_{i} y_{j}K(x_i,x_j)-\sum_{i=1}^{N} \alpha_{i}
$$
**这等价于经过映射函数$\phi$将原来的输入空间变换到一个新的特征空间，将输入空间中的内积$(x_i·x_j)$变换为特征空间中的内积$\phi(x_i)·\phi(x_j)$，在新的特征空间里从训练样本中学习线性支持向量机。当映射函数是非线性函数时，学习到的含有核函数的支持向量机是非线性分类模型。**

在这一过程中，学习是隐式地在特征空间进行的，不需要显式地定义特征空间和映射函数。
