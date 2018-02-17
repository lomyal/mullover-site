---
layout: post
title: 用ADMM实现统计学习问题的分布式计算
description: "此文可以当做 ADMM 的快速入门。交替方向乘子法（Alternating Direction Method of Multipliers，ADMM）是一种求解优化问题的计算框架, 适用于求解分布式凸优化问题，特别是统计学习问题。 ADMM 通过分解协调（Decomposition-Coordination）过程，将大的全局问题分解为多个较小、较容易求解的局部子问题，并通过协调子问题的解而得到大的全局问题的解。"
category: 机器学习
---

<div class="message">
  最近研读了 Boyd 2011 年那篇关于 ADMM 的综述。这是由两件事情促成的：一是刘鹏的《计算广告》中 CTR 预测技术那部分提到了用 ADMM 框架来并行化 LR 问题的求解；二是我的一个本科同学在读博士期间做的东西恰好用到了 ADMM，我们曾经谈到过这个框架。我从这篇综述里整理出了一个条思路，顺着这个思路看下去，就能对 ADMM 原理和应用有个大概的了解。因此，此文可以当做 ADMM 的快速入门。
</div>

交替方向乘子法（Alternating Direction Method of Multipliers，ADMM）是一种求解优化问题的计算框架, 适用于求解分布式凸优化问题，特别是统计学习问题。 ADMM 通过分解协调（Decomposition-Coordination）过程，将大的全局问题分解为多个较小、较容易求解的局部子问题，并通过协调子问题的解而得到大的全局问题的解。

ADMM 最早分别由 Glowinski & Marrocco 及 Gabay & Mercier 于 1975 年和 1976 年提出，并被 Boyd 等人于 2011 年重新综述并证明其适用于大规模分布式优化问题。由于 ADMM 的提出早于大规模分布式计算系统和大规模优化问题的出现，所以在 2011 年以前，这种方法并不广为人知。


## ADMM 计算框架

### 一般问题

若优化问题可表示为

\begin{equation}
\begin{aligned}
    \min \quad f(x) + g(z)\\
    \quad \quad \mathrm{s.t.} \quad Ax + Bz = c
\end{aligned}
\end{equation}

其中 \\( x \in R^{s}, z \in R^{n}, A \in R^{p \times s}, B \in R^{p \times n}, c \in R^{p}, f : R^{s} \rightarrow R, g : R^{n} \rightarrow R \\)。
\\(x\\) 与 \\(z\\) 是优化变量；\\( f(x) + g(z) \\) 是待最小化的目标函数（Objective Function），它由与变量 \\(x\\) 相关的 \\(f(x)\\) 和与变量 \\(x\\) 相关的 \\(g(z)\\) 这两部分构成，这种结构可以很容易地处理统计学习问题优化目标中的正则化项； \\(Ax + Bz = c\\) 是 \\(p\\) 个等式约束条件（Equality Constraints）的合写。其增广拉格朗日函数（Augmented Lagrangian）为

$$
L_{\rho}(x,z,y) = f(x) + g(z) + y^{T} (Ax + Bz - c) + (\rho / 2) \| Ax + Bz - c \| _{2}^{2}
$$

其中 \\(y\\) 是对偶变量（或称为拉格朗日乘子）， \\(\rho > 0\\) 是惩罚参数。 \\(L_{\rho}\\) 名称中的“增广”是指其中加入了二次惩罚项 \\((\rho / 2) \Vert Ax + Bz - c \Vert _{2}^{2}\\) 。

则该优化问题的 ADMM 迭代求解方法为

$$
\begin{aligned}
    x^{k + 1} &:= \arg\min_{x} L_{\rho}(x, z^{k}, y^{k}) \\
    z^{k + 1} &:= \arg\min_{z} L_{\rho}(x^{k + 1}, z, y^{k}) \\
    y^{k + 1} &:= y^{k} + \rho (Ax^{k + 1} + Bz^{k + 1} - c)
\end{aligned}
$$

令 \\( u = (1 / \rho) y \\) ，并对 \\( Ax + Bz - c \\) 配方，可得表示上更简洁的缩放形式（Scaled Form）

$$
\begin{aligned}
    x^{k + 1} &:= \arg\min_{x} \Big( f(x) + (\rho / 2) \| Ax + Bz^{k} - c + u^{k} \|_{2}^{2} \Big) \\
    z^{k + 1} &:= \arg\min_{z} \Big( g(z) + (\rho / 2) \| Ax^{k + 1} + Bz - c + u^{k} \|_{2}^{2} \Big) \\
    u^{k + 1} &:= u^{k} + Ax^{k + 1} + Bz^{k + 1} - c
\end{aligned}
$$

可以看出，每次迭代分为三步：

1. 求解与 \\(x\\) 相关的最小化问题，更新变量 \\(x\\)
2. 求解与 \\(z\\) 相关的最小化问题，更新变量 \\(z\\)
3. 更新对偶变量 \\(u\\)

ADMM名称中的“乘子法”是指这是一种使用增广拉格朗日函数（带有二次惩罚项）的对偶上升（Dual Ascent）方法，而“交替方向”是指变量 \\(x\\) 和 \\(z\\) 是交替更新的。两变量的交替更新是在 \\(f(x)\\) 或 \\(g(z)\\) 可分时可以将优化问题分解的关键原因。

### 收敛性

可以证明，当满足条件

1. 函数  \\(f, g\\) 具有 closed, proper, convex 的性质
2. 拉格朗日函数 \\(L_{0}\\) 有鞍点

时，ADMM 的迭代收敛（当  \\(k \rightarrow \infty \\) 时， \\(r^{k} \rightarrow 0, f(x^{k}) + g(z^{k}) \rightarrow p^{\star}, y^{k} \rightarrow y^{\star} \\) ）。这样的收敛条件比没有使用增广拉格朗日函数的对偶上升法的收敛条件宽松了不少。

在高精度要求下，ADMM 的收敛很慢；但在中等精度要求下，ADMM 的收敛速度可以接受（几十次迭代）。因此 ADMM 框架尤其适用于不要求高精度的优化问题，这恰好是大规模统计学习问题的特点。

### 一致性（Consensus）问题

一类可用 ADMM 框架解决的特殊优化问题是一致性（Consensus）问题，其形式为

$$
\min \quad \sum_{i = 1}^{N} f_{i}(z) + g(z)
$$

将加性优化目标 \\(\sum_{i = 1}^{N} f_{i}(z)\\) 转化为可分优化目标 \\(\sum_{i = 1}^{N} f_{i}(x_{i})\\) ，并增加相应的等式约束条件，可得其等价问题

\begin{equation}
\begin{aligned}
    \min \quad \sum_{i = 1}^{N} f_{i}(x_{i}) + g(z)\\
    \quad \quad \mathrm{s.t.} \quad x_{i} - z = 0, \quad i = 1, \dots, N
\end{aligned}
\end{equation}

这里约束条件要求每个子目标中的局部变量 \\(x_{i}\\) 与全局变量 \\(z\\) 一致，因此该问题被称为一致性问题。

可以看出，令式（1）中的  \\(x = (x_{1}^{T}, \dots, x_{N}^{T})^{T}, f(x) = \sum_{i = 1}^{N} f_{i}(x_{i}), A = I_{sN}, B = -(\underbrace{I_{s}, \dots, I_{s}}_{N})^{T}, c = 0\\) ，即得到式（2）。因此 Consensus 问题可用 ADMM 框架求解，其迭代方法为

$$
\begin{aligned}
    x^{k + 1}_{i} &:= \arg\min_{x_{i}} \Big( f_{i}(x_{i}) + (\rho / 2) \| x_{i} - z^{k} + u_{i}^{k} \|_{2}^{2} \Big) \\
    z^{k + 1} &:= \arg\min_{z} \Big( g(z) + (N \rho / 2) \| z - \overline{x}^{k + 1} - \overline{u}^{k} \|_{2}^{2} \Big) \\
    u^{k + 1}_{i} &:= u^{k}_{i} + x^{k + 1}_{i} - z^{k + 1}
\end{aligned}
$$

其中 \\(\overline{x} = (1/N) \sum_{i = 1}^{N} x_{i}, \overline{u} = (1/N) \sum_{i = 1}^{N} u_{i}\\) 。

可以看出，变量 \\(x\\) 和对偶变量 \\(u\\) 的更新都是可以采用分布式计算的。只有在更新变量 \\(z\\) 时，需要收集 \\(x\\) 和 \\(u\\) 分布式计算的结果，进行集中式计算。

## 统计学习问题应用

统计学习问题也是模型拟合问题，可表示为

$$
    \min \quad l(D,d,z) + r(z)
$$

其中  \\(z \in R^{n}\\) 是待学习的参数，  \\(D \in R^{m \times n}\\) 是模型的输入数据集，  \\(d \in R^{m}\\) 是模型的输出数据集，  \\(l : R^{m \times n} \times R^{m} \times R^{n} \rightarrow R \\) 是损失函数， \\(r : R^{n} \rightarrow R \\) 是正则化项， \\(m\\) 表示数据的个数， \\(n\\) 表示特征的个数。

对于带L1正则化项的线性回归（Lasso），其平方损失函数为

$$
l(D,d,z) = (1/2)\|Dz - d\|^{2}_{2}
$$

对于逻辑回归（Logistic Regression），其极大似然损失函数为

$$
l(D,d,z) = \textbf{1}^{T} \Big( \log \big( \exp (Dz) + \textbf{1} \big) - Dzd^{T} \Big)
$$

对于线性支持向量机（Linear Support Vector Machine），其合页（Hinge）损失函数为

$$
l(D,d,z) = \textbf{1}^{T}(\textbf{1} - Dzd^{T})_{+}
$$

将训练数据集（输入数据和输出数据）在样本的维度（ \\(m\\) ）划分成 \\(N\\) 块

$$
D =
\begin{pmatrix}
    D_{1} \\
    \vdots \\
    D_{N}
\end{pmatrix},
d =
\begin{pmatrix}
    d_{1} \\
    \vdots \\
    d_{N}
\end{pmatrix},
$$

其中 \\(D_{i} \in R^{m_{i} \times n}, d_{i} \in R^{m_{i}}, \sum_{i = 1}^{N} m_{i} = m\\) ，若有局部损失函数 \\(l_{i} : R^{m_{i} \times n} \times R^{m_{i}} \times R^{n} \rightarrow R \\) ，可得

\begin{equation}
\begin{aligned}
    \min \quad \sum_{i = 1}^{N} l_{i}(D_{i}, d_{i}, x_{i}) + r(z)\\
    \quad \quad \mathrm{s.t.} \quad x_{i} - z = 0, \quad i = 1, \dots, N
\end{aligned}
\end{equation}

可以看出，令式（2）中的  \\(f_{i}(x_{i}) = l_{i}(D_{i}, d_{i}, x_{i}), g(z) = r(z)\\) ，即得到式（3），因此
统计学习问题可用 Consensus ADMM 实现分布式计算，其迭代方法为

$$
\begin{aligned}
    x^{k + 1}_{i} &:= \arg\min_{x_{i}} \Big( l_{i}(D_{i}, d_{i}, x_{i}) + (\rho / 2) \| x_{i} - z^{k} + u_{i}^{k} \|_{2}^{2} \Big) \\
    z^{k + 1} &:= \arg\min_{z} \Big( r(z) + (N \rho / 2) \| z - \overline{x}^{k + 1} - \overline{u}^{k} \|_{2}^{2} \Big) \\
    u^{k + 1}_{i} &:= u^{k}_{i} + x^{k + 1}_{i} - z^{k + 1}
\end{aligned}
$$

## 分布式实现

### MPI

MPI 是一个语言无关的并行算法消息传递规约。使用 MPI 范式的 Consensus ADMM 算法如下所示。

<!-- \begin{algorithm}
    \caption{Consensus ADMM in MPI}\label{alg_mpi}
    \begin{algorithmic}[1]
        \State \textbf{initialize} $N$ processes, along with $x_{i}, u_{i}, r_{i}, z$
        \Repeat
            \State Update $ r_{i} = x_{i} - z $
            \State Update $ u_{i} := u_{i} + x_{i} - z $
            \State Update $ x_{i} := \arg\min_{x} \big( f_{i}(x) + (\rho / 2) \| x - z + u_{i} \|_{2}^{2} \big) $
            \State Let $ w := x_{i} + u_{i} $ and $ t := \| r_{i} \|_{2}^{2} $
            \State $Allreduce$ $w$ and $t$
            \State Let $ z^{\mathrm{prev}} := z $ and update $z := \arg\min_{z} \big( g(z) + (N \rho / 2) \| z - w / N \|_{2}^{2} \big)$
        \Until {$ \rho \sqrt{N} \| z - z^{\mathrm{prev}} \|_{2} \le \epsilon^{\mathrm{conv}} $ and $ \sqrt{t} \le \epsilon^{\mathrm{feas}} $}
    \end{algorithmic}
\end{algorithm} -->

> 1. **Initialize** \\(N\\) processes, along with \\(x_{i}, u_{i}, r_{i}, z\\)
>
> 2. **Repeat**
>
> 3. &nbsp;&nbsp;&nbsp; Update \\( r_{i} = x_{i} - z \\)
>
> 4. &nbsp;&nbsp;&nbsp; Update \\( u_{i} := u_{i} + x_{i} - z \\)
>
> 5. &nbsp;&nbsp;&nbsp; Update \\( x_{i} := \arg\min_{x} \big( f_{i}(x) + (\rho / 2) \Vert x - z + u_{i} \Vert_{2}^{2} \big) \\)
>
> 6. &nbsp;&nbsp;&nbsp; Let \\( w := x_{i} + u_{i} \\) and \\( t := \Vert r_{i} \Vert_{2}^{2} \\)
>
> 7. &nbsp;&nbsp;&nbsp; *Allreduce* \\(w\\) and \\(t\\)
>
> 8. &nbsp;&nbsp;&nbsp; Let \\( z^{\mathrm{prev}} := z \\) and update \\(z := \arg\min_{z} \big( g(z) + (N \rho / 2) \| z - w / > N \|_{2}^{2} \big)\\)
>
> 9. **Until** \\( \rho \sqrt{N} \Vert z - z^{\mathrm{prev}} \Vert_{2} \le \epsilon^{\mathrm{conv}} \\) and \\( \sqrt{t} \le \epsilon^{\mathrm{feas}} \\)

该算法中假设有 \\(N\\) 个处理器，每个处理器都运行同样的程序，只是处理的数据不同。第6步中的 *Allreduce* 是 MPI 中定义的操作，表示对相应的局部变量进行全局操作（如这里的求和操作），并将结果更新到每一个处理器。

### MapReduce

MapReduce 是一个在工业界和学术界都很流行的分布式批处理编程模型。使用 MapReduce 范式的 Consensus ADMM 算法（一次迭代）如下所示。

<!-- \begin{algorithm}
    \caption{Consensus ADMM in MapReduce}\label{alg_mapreduce}
    \begin{algorithmic}[1]
        \Function{map}{key $i$, dataset $D_{i}$}
            \State Read $(x_{i}, u_{i}, \hat{z})$ from distributed database
            \State Compute $z := \arg\min_{z} \big( g(z) + (N \rho / 2) \| z - \hat{z} / N \|_{2}^{2} \big)$
            \State Update $ u_{i} := u_{i} + x_{i} - z $
            \State Update $ x_{i} := \arg\min_{x} \big( f_{i}(x) + (\rho / 2) \| x - z + u_{i} \|_{2}^{2} \big) $
            \State $Emit$ (key \textsc{central}, record $(x_{i}, u_{i}))$)
        \EndFunction
        \Function{reduce}{key \textsc{central}, records $(x_{1}, u_{1}), \dots, (x_{N}, u_{N})$}
            \State Update $ \hat{z} := \sum_{i = 1}^{N} (x_{i} + u_{i}) $
            \State $Emit$ (key $j$, record $(x_{j}, u_{j}, z) $) to distributed database for $j = 1, \dots, N$
        \EndFunction
    \end{algorithmic}
\end{algorithm} -->

> **Function** map(key \\(i\\) , dataset \\(D_{i}\\) )
>
> 1. Read \\((x_{i}, u_{i}, \hat{z})\\) from distributed database
>
> 2. Compute \\(z := \arg\min_{z} \big( g(z) + (N \rho / 2) \| z - \hat{z} / N \|_{2}^{2} \big)\\)
>
> 3. Update \\( u_{i} := u_{i} + x_{i} - z \\)
>
> 4. Update \\( x_{i} := \arg\min_{x} \big( f_{i}(x) + (\rho / 2) \| x - z + u_{i} \|_{2}^{2} \big) \\)
>
> 5. *Emit* (key \\(\small{CENTRAL}\\) , record \\((x_{i}, u_{i})\\) )
>
> **EndFunction**
>
> **Function** reduce (key \\(\small{CENTRAL}\\) , records \\((x_{1}, u_{1}), \dots, (x_{N}, u_{N})\\) )
>
> 1. Update \\( \hat{z} := \sum_{i = 1}^{N} (x_{i} + u_{i}) \\)
>
> 1. *Emit* (key \\(j\\) , record \\((x_{j}, u_{j}, z) \\) ) to distributed database for \\(j = 1, \dots, N\\)
>
> **EndFunction**


为了实现多次迭代，该算法需要由一个 wrapper 程序在每次迭代结束后判断是否满足迭代终止条件 \\(\rho \sqrt{N} \Vert z - z^{\mathrm{prev}} \Vert_{2} \le \epsilon^{\mathrm{conv}}\\) 且 \\(\big(\sum_{i = 1}^{N} \\| x_{i} - z \\| _{2}^{2} \big)^{1/2} \le \epsilon^{\mathrm{feas}}\\) ，若不满足则启动下一次迭代。

## 参考文献

- Boyd S, Parikh N, Chu E, et al. [Distributed optimization and statistical learning via the alternating direction method of multipliers](https://www.researchgate.net/profile/Borja_Peleato/publication/220416607_Distributed_Optimization_and_Statistical_Learning_via_the_Alternating_Direction_Method_of_Multipliers/links/00463534406c9dfd48000000.pdf)[J]. Foundations and Trends® in Machine Learning, 2011, 3(1): 1-122.

- Eckstein J, Yao W. [Understanding the convergence of the alternating direction method of multipliers: Theoretical and computational perspectives](http://www.optimization-online.org/DB_FILE/2015/06/4954.pdf)[J]. Pac. J. Optim., 2014.

- Lusk E, Huss S, Saphir B, et al. [MPI: A Message-Passing Interface Standard Version 3.1](http://www.mpi-forum.org/docs/mpi-3.1/mpi31-report.pdf)[J], 2015.

- Dean J, Ghemawat S. [MapReduce: simplified data processing on large clusters](http://www.cs.ucsb.edu/~cappello/290B/papers/MapReduce/mapreduce-osdi04.pdf)[J]. Communications of the ACM, 2008, 51(1): 107-113.