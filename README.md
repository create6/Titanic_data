### 人工智能概述

使用机器模仿人类学习和其他方面的智能

- 图灵测试

测试者与被测试者（一个人和一台机器）隔开的情况下，通过一些装置（如键盘）向被测试者随意提问。

多次测试（一般为**5min**之内），如果有超过**30%**的测试者不能确定被测试者是人还是机器，那么这台机器就通过了测试，并被认为具有**人类智能**。

- 主要分支

  - 计算机视觉(图像形成,图像处理,图像提取,图像三维推理)
  - 自然语言处理(文本挖掘与分类,机器翻译,语音识别)
  - 机器人

- 三要素

  - 数据
  - 算法
  - 计算力






### 机器学习库

#### Numpy

- 了解Numpy运算速度上的优势
- 知道数组的属性，形状、类型
- 应用Numpy实现数组的基本操作
- 应用随机数组的创建实现正态分布应用
- 应用Numpy实现数组的逻辑运算
- 应用Numpy实现数组的统计运算
- 应用Numpy实现数组之间的运算



##### ndarray到底跟原生python列表有什么不同呢

ndarray在存储数据的时候，数据与数据的地址都是连续的，这样就给使得批量操作数组元素时速度更快。

这是因为ndarray中的所有元素的类型都是相同的，而Python列表中的元素类型是任意的，所以ndarray在存储元素时内存可以连续，而python原生list就只能通过寻址方式找到下一个元素，这虽然也导致了在通用性能方面Numpy的ndarray不及Python原生list，但在科学计算中，Numpy的ndarray就可以省掉很多循环语句，代码使用方面比Python原生list简单的多。

数据类型必须相同




##### ndarray的属性

数组属性反映了数组本身固有的信息。

| 属性名字         | 属性解释                   |
| ---------------- | -------------------------- |
| ndarray.shape    | 数组维度的元组             |
| ndarray.ndim     | 数组维数                   |
| ndarray.size     | 数组中的元素数量           |
| ndarray.itemsize | 一个数组元素的长度（字节） |
| ndarray.dtype    | 数组元素的类型             |







#### Pandas

- 了解Numpy与Pandas的不同
- 说明Pandas的Series与Dataframe两种结构的区别
- 了解Pandas的MultiIndex与panel结构
- 应用Pandas实现基本数据操作
- 应用Pandas实现数据的统计分析
- 应用Pandas实现数据的逻辑筛选
- 应用Pandas实现数据的算数运算
- 应用Pandas实现数据的缺失值处理
- 应用Pandas实现数据的离散化处理
- 应用Pandas实现数据的合并
- 应用crosstab和pivot_table实现交叉表与透视表
- 应用groupby和聚合函数实现数据的分组与聚合
- 了解Pandas的plot画图功能
- 应用Pandas实现数据的读取和存储



##### Dataframe常用操作

df.info()  打印二维数组的信息

df.describe（）  查看数据值列的汇总统计

可返回变量和观测的数量、缺失值和唯一值的数目、平均值、分位数等相关信息

df.T  index 跟 columns 对调

df.columns求列

df.index 求行



##### 薄弱点

回顾pandas 切片

DataFrame 切片

 报错：

```
'DataFrame' object has no attribute 'type'
'RangeIndex' object has no attribute 'index'
```

##### [pandas (loc、iloc、ix)的区别](https://www.cnblogs.com/keye/p/7825280.html)

**loc：**通过行标签索引数据

**iloc：**通过行号索引行数据

**ix：**通过行标签或行号索引数据（基于loc和iloc的混合）

 

使用loc、iloc、ix索引第一行数据：

**loc：**

**![img](https://img2018.cnblogs.com/blog/1235684/201903/1235684-20190314193022370-1097142667.png)**

![img](https://images2017.cnblogs.com/blog/1235684/201711/1235684-20171113103030031-203044434.png)

![img](https://images2017.cnblogs.com/blog/1235684/201711/1235684-20171113103306906-486702850.png)

![img](https://images2017.cnblogs.com/blog/1235684/201711/1235684-20171113103325702-1905964753.png)

**iloc：**

![img](https://images2017.cnblogs.com/blog/1235684/201711/1235684-20171113103846452-166603526.jpg)

![img](https://images2017.cnblogs.com/blog/1235684/201711/1235684-20171113104105999-581459458.jpg)

![img](https://images2017.cnblogs.com/blog/1235684/201711/1235684-20171113104113327-2047599815.jpg)

**ix：**

![img](https://images2017.cnblogs.com/blog/1235684/201711/1235684-20171113104518140-1699869554.jpg)

 

#### Matplotlib

- 知道Matplotlib的架构
- 应用Matplotlib的基本功能实现图形显示
- 应用Matplotlib实现多图显示
- 应用Matplotlib实现不同画图种类

##### 常用图

- 1折线图: 

     概念: 用于展示数据的变化情况的

    	API: plt.plot(x, y)

- 2散点图: 用于分析两个变量的规律, 展示离散点分布情况

     API: plt.scatter(x, y)

- 3柱状图: 统计,对比,离散

     API: plt.bar(x, height, width, color)

     x : x轴的标量序列

     height: 标量或标量序列, 柱状图的高度,或者为应变量

     width : 柱状图的宽度, 默认值0.8

     align : 柱状图在x维度上的对齐方式, {‘center’, ‘edge’}, 可选, 默认: ‘center’

     **kwargs :

     color:选择柱状图的颜色

```python
import matplotlib.pyplot as plt
#准备数据
name = ['雷神3：诸神黄昏','正义联盟','东方快车谋杀案','寻梦环游记','全球风暴', '降魔传','追捕','七十七天','密战','狂兽','其它']
income = [73853,57767,22354,15969,14839,8725,8716,8318,7916,6764,52222]
#创建画布
plt.figure(figsize=(20,8),dpi=100)
#绘制柱状图
# x = range(len(name))
x =name
plt.bar(x,income,width=0.5,color=['y','b','r','k'])
#设置x轴参数
plt.xticks(x,name)
plt.show()

# 设置刻度字体大小
# plt.tick_params(labelsize=20)   
```



- 4直方图: 展示连续数据的分布情况

     API: plt.hist(x, bins)

        x : 数组或数组的序列, 表示要展示的数据

        bins : 整数,序列 可选

        如果是整数就是柱状体的个数

        如果序列就是每个柱状体的边缘值, 左开右闭.

```python
x2 = np.random.normal(loc=2,scale=4,size=100000)
#loc均值,scale 标准差
#画布
plt.figure(figsize=(20,8),dpi=100)
plt.hist(x2,bins=1000)
plt.show()
```



- 5饼状图: 占比

    API: plt.pie(x, labels, autopct, colors)

       x:数量，自动算百分比

       labels:每部分名称

       autopct:占比显示指定  '%.2f%%'

       colors:每部分颜色

```python
import matplotlib.pyplot as plt
import numpy as np
#准备数据
x = ['周一','周二','周三','周四','周五','周六','周日']
y = [12,23,31,44,52,65,79]
#创建画布
plt.figure(figsize=(20,8),dpi=100)
plt.pie(y,labels=x,autopct='%.2f%%')
#绘图
plt.show()
```





#### Seaborn

Seaborn是基于matplotlib的Python可视化库。 它提供了一个高级界面来绘制有吸引力的统计图形。Seaborn其实是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，不需要经过大量的调整就能使你的图变得精致。**但应强调的是，应该把Seaborn视为matplotlib的补充，而不是替代物。**



##### kdeplot(核密度估计图)

核密度估计(kernel density estimation)是在**概率论**中用来估计未知的**密度函数**，属于非参数检验方法之一。通过核密度估计图可以比较直观的看出数据样本本身的分布特征。

##### distplot

displot()集合了matplotlib的hist()与核函数估计kdeplot的功能，增加了rugplot分布观测条显示与利用scipy库fit拟合参数分布的新颖用途。

```python
import matplotlib.pyplot as plt
import seaborn as sns
# 绘制图像
plt.figure(figsize=(12,4))
sns.distplot(train['longestKill'], bins=10)
plt.show()
```

##### countplot 计数直方图
countplot 故名思意，是“计数图”的意思，可将它认为一种应用到分类变量的直方图，也可认为它是用以比较类别间计数差，调用 count 函数的 barplot；

countplot 参数和 barplot 基本差不多，可以对比着记忆，有一点不同的是 countplot 中不能同时输入 x 和 y ，且 countplot 没有误差棒。



```python
# 首先绘制玩家杀敌数的条形图
plt.figure(figsize=(10,4))
sns.countplot(data=train, x=train['kills']).set_title('Kills')
plt.show()
```

#### Sklearn

- Python语言的机器学习工具
- Scikit-learn包括许多知名的机器学习算法的实现
- Scikit-learn文档完善，容易上手，丰富的API
- 目前稳定版本0.19.1



##### classification分类

常用的分类：线性、决策树、SVM、KNN，朴素贝叶斯；集成分类：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees

##### regressor 回归

常用的回归：线性、决策树、SVM、KNN ；集成回归：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees

#from sklearn.linear_model import LinearRegression,SGDRegressor 
#线性回归模型与梯度下降模型(随机梯度下降算法)

##### clustering 聚类

常用聚类：k均值（K-means）、层次聚类（Hierarchical clustering）、DBSCAN

##### Dimensionality reduction 降维

常用降维：LinearDiscriminantAnalysis、PCA

##### Model selection 模型选择      

#from sklearn.model_selection import train_test_split,GridSearchCV 数据集分割,网格搜索与交叉验证

##### Preprocessing (特征)预处理   

#from sklearn.preprocessing import StandardScaler,MinMaxScaler 标准化与归一化



#####  sklearn模型的保存和加载API

- from sklearn.externals import joblib
  - 保存：joblib.dump(estimator, 'test.pkl')
  - 加载：estimator = joblib.load('test.pkl')
  - 注意:  文件名的后缀是 `pkl



##### 分类与回归

分类和回归的本质是一样的，都是**对输入做出预测**，其区别在于输出的类型。

分类问题：分类问题的输出是**离散型变量**(如: +1、-1)，是一种**定性输出**。(预测明天天气是阴、晴还是雨) 
回归问题：回归问题的输出是**连续型变量**，是一种**定量输出**。(预测明天的温度是多少度)。



##### 模型选择



##### 模型优化



#### Surprise

Surprise · A Python scikit for recommender systems.

<https://github.com/NicolasHug/Surprise>

<http://surpriselib.com/>

协同过滤

学习内容主要参考唐宇迪教程与官方文档





### 机器学习步骤

#### 1 抽象成数学问题

明确问题是进行机器学习的第一步。机器学习的训练过程通常都是一件非常耗时的事情，胡乱尝试时间成本是非常高的。

这里的抽象成数学问题，指的明确我们可以获得什么样的数据，抽象出的问题，是一个分类还是回归或者是聚类的问题。

#### 2 获取数据

数据决定了机器学习结果的上限，而算法只是尽可能逼近这个上限。

数据要有代表性，否则必然会过拟合。

而且对于分类问题，数据偏斜不能过于严重，不同类别的数据数量不要有数量级的差距。

而且还要对数据的量级有一个评估，多少个样本，多少个特征，可以估算出其对内存的消耗程度，判断训练过程中内存是否能够放得下。如果放不下就得考虑改进算法或者使用一些降维的技巧了。如果数据量实在太大，那就要考虑分布式了。

#### 3 特征预处理与特征选择

良好的数据要能够提取出良好的特征才能真正发挥作用。

特征预处理、数据清洗是很关键的步骤，往往能够使得算法的效果和性能得到显著提高。归一化、离散化、因子化、缺失值处理、去除共线性等，数据挖掘过程中很多时间就花在它们上面。这些工作简单可复制，收益稳定可预期，是机器学习的基础必备步骤。

筛选出显著特征、摒弃非显著特征，需要机器学习工程师反复理解业务。这对很多结果有决定性的影响。特征选择好了，非常简单的算法也能得出良好、稳定的结果。这需要运用特征有效性分析的相关技术，如相关系数、卡方检验、平均互信息、条件熵、后验概率、逻辑回归权重等方法。

#### 数据处理!!

数据特征处理:

- 归一化

- 标准化

- 离散化(eg.sigmoid二分类,)



#### 4 训练模型与调优

直到这一步才用到我们上面说的算法进行训练。现在很多算法都能够封装成黑盒供人使用。但是真正考验水平的是调整这些算法的（超）参数，使得结果变得更加优良。这需要我们对算法的原理有深入的理解。理解越深入，就越能发现问题的症结，提出良好的调优方案。

#### 5 模型诊断

如何确定模型调优的方向与思路呢？这就需要对模型进行诊断的技术。

过拟合、欠拟合 判断是模型诊断中至关重要的一步。常见的方法如交叉验证，绘制学习曲线等。过拟合的基本调优思路是增加数据量，降低模型复杂度。欠拟合的基本调优思路是提高特征数量和质量，增加模型复杂度。

误差分析 也是机器学习至关重要的步骤。通过观察误差样本全面分析产生误差的原因:是参数的问题还是算法选择的问题，是特征的问题还是数据本身的问题……

诊断后的模型需要进行调优，调优后的新模型需要重新进行诊断，这是一个反复迭代不断逼近的过程，需要不断地尝试， 进而达到最优状态。

#### 6 模型融合

一般来说，模型融合后都能使得效果有一定提升。而且效果很好。

工程上，主要提升算法准确度的方法是分别在模型的前端（特征清洗和预处理，不同的采样模式）与后端（模型融合）上下功夫。因为他们比较标准可复制，效果比较稳定。而直接调参的工作不会很多，毕竟大量数据训练起来太慢了，而且效果难以保证。

#### 7 上线运行

这一部分内容主要跟工程实现的相关性比较大。工程上是结果导向，模型在线上运行的效果直接决定模型的成败。 不单纯包括其准确程度、误差等情况，还包括其运行的速度(时间复杂度)、资源消耗程度（空间复杂度）、稳定性是否可接受。

这些工作流程主要是工程实践上总结出的一些经验。并不是每个项目都包含完整的一个流程。这里的部分只是一个指导性的说明，只有大家自己多实践，多积累项目经验，才会有自己更深刻的认识。




### 机器学习算法分类

#### 监督学习

      输入数据: 特征 和 目标值
      算法: 
        目标值连续: 回归问题		线性回归	 	  	
        目标值离散: 分类问题		KNN

#### 无监督学习

      数据数据: 只有特征值没有目标值
      算法: 聚类

#### 半监督学习

      输入数据: 特征 + 目标值(部分)

#### 强化学习

    	概念:  智能体(Agent)不断与环境进行交互, 通过试错得到最佳策略. 

```
四要素:
智能体(Agent), 环境(Environment), 行为(Action), 奖励(Reward)
```



### KNN(K-近邻算法)

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#加载数据
iris =load_iris()

#数据基本处理,数据分割
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=22)

#特征工程(特征预处理-标准化),对训练集与测试集的特征值进行标准化
transfer =StandardScaler()
x_train =transfer.fit_transform(x_train)
#x_test=transfer.fit_transform(x_test)
x_test=transfer.transform(x_test) #使用上次的均值与标准差

#机器学习,模型训练
estimator=KNeighborsClassifier(n_neighbors=5)

#训练集特征值,训练集的目标值,x为特征值,y为目标值
estimator.fit(x_train,y_train)

#模型评估
#使用模型对测试集进行预测
y_pre=estimator.predict(x_test)
print(y_pre)
print(y_pre == y_test)

#准确率
score=estimator.score(x_test,y_test)
print('准确率:',score)
```



- 优点: 
  ​    简单有效
  ​    重新训练的代价低
  ​    适合类域交叉样本
  ​    适合大样本自动分类
- 缺点:
  ​    惰性学习
  ​    类别评分不是规格化
  ​    输出可解释性不强
  ​    对不均衡的样本不擅长(用归一化,标准化来处理)
  ​    计算量较大



##### K近邻算法中的k值的选取对算法有没有影响？

- **K值过小**：

  容易受到异常点的影响

- k值过大：

  受到样本均衡的问题



在实际应用中，K值一般取一个比较小的数值，例如采用交叉验证法（简单来说，就是把训练数据在分成两组:训练集和验证集）来选择最优的K值。对这个简单的分类器进行泛化，用核方法把这个线性模型扩展到非线性的情况，具体方法是把低维数据集映射到高维特征空间。

网格搜索:通常情况下，**有很多参数是需要手动指定的（如k-近邻算法中的K值），这种叫超参数**。但是手动过程繁杂，所以需要对模型预设几种超参数组合。**每组超参数都采用交叉验证来进行评估。最后选出最优参数组合建立模型。**



#### 归一化与标准化

##### 归一化表达式:

$$
X{'}=\frac{x-min}{max-min}  
$$
$$
XX{'}=X{'}*(mx-mi)+mi
$$



##### 标准化表达式:

$$
X{'}=\frac{x-mean}{σ}
$$




- 通常使用标准化
- 为什么异常值对归一化的影响很大，请简要说明

由其计算公式可知,一组数据中的极值对结果有决定作用,当一组数据中有异常值,通常是偏大或者偏小,既会归一化结果.

﻿最大值最小值是变化的，另外，最大值与最小值非常容易受异常点影响，**所以这种方法鲁棒性较差，只适合传统精确小数据场景。**



#### 交叉验证

- 目的: 为了提高模型训练结果可信度.
- 防止特殊数据集中在一个区域




```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV#交叉验证
from sklearn.preprocessing import StandardScaler  #标准化
from sklearn.neighbors import KNeighborsClassifier

#1,加载数据集
iris=load_iris()

#2,数据基本处理(分割数据集)
#random_state: 随机数种子, 如果种子相同分割数据集就相同, 如果不同分割数据集也不同
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.3,random_state=8) 

#3,特征工程(特征处理-标准化)
transfer = StandardScaler()
x_train =transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

#4,机器学习(模型训练):K近邻
estimator = KNeighborsClassifier()

#网格搜索和交叉验证进行参数调优
#超参数数字典
param_grid ={'n_neighbors':[3,5,7,9]}
estimator=GridSearchCV(estimator,param_grid=param_grid,cv=5)

#学习:训练集特征值,训练集目标值
estimator.fit(x_train,y_train)
#5,模型评估
#使用模型对测试集进行预测,参数为测试集的特征值,返回为预测目标值
y_pre =estimator.predict(x_test)

#准确率
score =estimator.score(x_test,y_test)
print('准确率:',score)
#风格搜索与交叉验证结果
print('交叉验证最好分数:',estimator.best_score_)
print('交叉验证最好的模型:',estimator.best_estimator_) #最佳模型与老师的不同
print('交叉验证的结果:\n',estimator.cv_results_)
```




### 线性回归







#### 似然函数,对数似然

- 让预测值越接近真实值(极大似然),让似然函数越大越好,减号右边的式子越小越好,得到最小二乘法



#### 通用公式cz

线性回归(Linear regression)是利用**回归方程(函数)**对**一个或多个自变量(特征值)和因变量(目标值)之间**关系进行建模的一种分析方式。



####  损失函数



- yi为第i个训练样本的真实值
- h(xi)为第i个训练样本特征值组合预测函数
- 又称最小二乘法

如何去减少这个损失，使我们预测的更加准确些？既然存在了这个损失，我们一直说机器学习有自动学习的功能，在线性回归这里更是能够体现。这里可以通过一些优化方法去优化（其实是数学当中的求导功能）回归的总损失



#### 优化算法

- 如何去求模型当中的θ，使得损失最小？（目的是找到最小损失对应的θ值）



  ##### 线性回归经常使用的两种优化算法





- [x] 1.正规方程: 根据样本数据直接计算一个最好的模型系数和偏置(**计算量大**)




		(1)中 要消掉X,X不一定是方阵,先转置,再乘以X的逆来消除X



- [x] 2.梯度下降: 从一个任意的模型系数和偏置开始, 一步一步进行优化, 最终得到一个最好的模型系数和偏置.	





#### 梯度下降!!



##### 用小的学习率,用大的迭代率



- 单变量





- **多变量**

  - 我们假设有一个目标函数 ：:J(θ) = θ12 + θ22

    现在要通过梯度下降法计算这个函数的最小值。我们通过观察就能发现最小值其实就是 (0，0)点。但是接下 来，我们会从梯度下降算法开始一步步计算到这个最小值! 我们假设初始的起点为: θ0 = (1, 3)

    初始的学习率为:α = 0.1

    函数的梯度为:▽:J(θ) =< 2θ1 ,2θ2>



- α是什么含义？

α在梯度下降算法中被称作为**学习率**或者**步长**，意味着我们可以通过α来控制每一步走的距离，以保证不要步子跨的太大扯着蛋，其实就是不要走太快，错过了最低点。同时也要保证不要走的太慢，导致太阳下山了，还没有走到山下。所以α的选择在梯度下降法中往往是很重要的！α不能太大也不能太小，太小的话，可能导致迟迟走不到最低点，太大的话，会导致错过最低点！



- 为什么梯度要乘以一个负号？

梯度前加一个负号，就意味着朝着梯度相反的方向前进！我们在前文提到，梯度的方向实际就是函数在此点上升最快的方向！而我们需要朝着下降最快的方向走，自然就是负的梯度的方向，所以此处需要加上负号



##### 梯度下降和正规方程的对比

| 梯度下降             | 正规方程                        |
| :------------------- | :------------------------------ |
| 需要选择学习率       | 不需要                          |
| 需要迭代求解         | 一次运算得出                    |
| 特征数量较大可以使用 | 需要计算方程，时间复杂度高O(n3) |



##### 梯度下降算法

- 全梯度下降算法FGD (Full gradient descent），
- 随机梯度下降算法SGD（Stochastic gradient descent），求解速度最快,当遇到噪声时会陷入局部最优解
- 随机平均梯度下降算法SAGD（Stochastic average gradient descent）
- 小批量梯度下降算法（Mini-batch gradient descent）,

它们都是为了正确地调节权重向量，通过为每个权重计算一个梯度，从而更新权值，使目标函数尽可能最小化。其差别在于样本的使用方式不同。



##### 总结

**所以有了梯度下降这样一个优化算法，回归就有了"自动学习"的能力**

**（1）FG方法由于它每轮更新都要使用全体数据集，故花费的时间成本最多，内存存储最大。**

**（2）SAG在训练初期表现不佳，优化速度较慢。这是因为我们常将初始梯度设为0，而SAG每轮梯度更新都结合了上一轮梯度值。**

**（3）综合考虑迭代次数和运行时间，SG表现性能都很好，能在训练初期快速摆脱初始梯度值，快速将平均损失函数降到很低。但要注意，在使用SG方法时要慎重选择步长，否则容易错过最优解。**

**（4）mini-batch结合了SG的“胆大”和FG的“心细”，从6幅图像来看，它的表现也正好居于SG和FG二者之间。在目前的机器学习领域，mini-batch是使用最多的梯度下降算法，正是因为它避开了FG运算效率低成本大和SG收敛效果不稳定的缺点。**





#### 案例：波士顿房价预测

- general(LinearRegression)     #线性回归
- general(SGDRegressor)          #随机梯度下降法
- general(Ridge)                         #随机平均梯度下降法,L2正则化项

```python
from sklearn.datasets import load_boston  #数据
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge 
#线性回归模型与梯度下降模型(随机梯度下降算法),岭回归
from sklearn.metrics import mean_squared_error,mean_absolute_error    
#mean_squared_error均方误差,mean_absolute_error平均均方误差


def general(estimator_method):
    #加载数据
    data = load_boston()
    #数据分割
    x_train,x_test,y_train,y_test =train_test_split(data.data,data.target,test_size=0.25,random_state=8)
    #特征工程,标准化
    transfer =StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.fit_transform(x_test)
    #机器学习 
    estimator = estimator_method()#代入
    estimator.fit(x_train,y_train)
    
    #模型评估
    print('模型系数:',estimator.coef_)
    print('模型偏置:',estimator.intercept_)
    #预测
    y_pre = estimator.predict(x_test)

    #模型评估
    mse =mean_squared_error(y_test,y_pre)
    print('均方误差:',mse)
    #平均绝对误差
    mae = mean_absolute_error(y_test,y_pre)
    print('平均绝对误差:',mae)
    print('-'*20)
    
# 调用   
general(LinearRegression)
general(SGDRegressor)
general(Ridge)




模型系数: [-0.98162265  1.16064607  0.18611408  0.64865713 -1.48273565  2.67325335
 -0.16756838 -3.00571558  2.29915542 -1.83639913 -1.92095414  0.85800075
 -4.05354071]
模型偏置: 22.52163588390508
均方误差: 22.231973959150817
平均绝对误差: 3.206908326834049
--------------------
模型系数: [-0.75213377  0.66397271 -0.26551159  0.77079673 -0.60301087  3.20383138
 -0.27078881 -1.77859629  0.67558832 -0.42733849 -1.8114033   0.82617229
 -3.66771429]
模型偏置: [22.06478046]
均方误差: 23.938544374219894
平均绝对误差: 3.2705445876283457
--------------------
模型系数: [-0.97246454  1.14327275  0.15848304  0.65305661 -1.45541569  2.68212945
 -0.17139627 -2.97390427  2.22587256 -1.76604839 -1.91302371  0.8558563
 -4.03757414]
模型偏置: 22.52163588390508
均方误差: 22.24949018124456
平均绝对误差: 3.2059213973436536
```


#### 过拟合与欠拟合

- 过拟合：一个假设**在训练数据上能够获得比其他假设更好的拟合， 但是在测试数据集上却不能很好地拟合数据**，此时认为这个假设出现了过拟合的现象。(模型过于复杂)
- 欠拟合：一个假设**在训练数据上不能获得更好的拟合，并且在测试数据集上也不能很好地拟合数据**，此时认为这个假设出现了欠拟合的现象。(模型过于简单)

简单描述欠拟合和过拟合出现的原因及其解决方法

- 欠拟合原因以及解决办法
  - 原因：学习到数据的特征过少
  - 解决办法：
    - **1）添加其他特征项，**有时候我们模型出现欠拟合的时候是因为特征项不够导致的，可以添加其他特征项来很好地解决。例如，“组合”、“泛化”、“相关性”三类特征是特征添加的重要手段，无论在什么场景，都可以照葫芦画瓢，总会得到意想不到的效果。除上面的特征之外，“上下文特征”、“平台特征”等等，都可以作为特征添加的首选项。
    - **2）添加多项式特征**，这个在机器学习算法里面用的很普遍，例如将线性模型通过添加二次项或者三次项使模型泛化能力更强。
- 过拟合原因以及解决办法
  - 原因：原始特征过多，存在一些嘈杂特征， 模型过于复杂是因为模型尝试去兼顾各个测试数据点
  - 解决办法：
    - 1）重新清洗数据，导致过拟合的一个原因也有可能是数据不纯导致的，如果出现了过拟合就需要我们重新清洗数据。
    - 2）增大数据的训练量，还有一个原因就是我们用于训练的数据量太小导致的，训练数据占总数据的比例过小。
    - **3）正则化**
    - 4）减少特征维度，防止**维灾难**



#### 正则化(处理过拟合)

在解决回归过拟合中，我们选择正则化。但是对于其他机器学习算法如分类算法来说也会出现这样的问题，除了一些算法本身作用之外（决策树、神经网络），我们更多的也是去自己做特征选择，包括之前说的删除、合并一些特征

- L2正则化(正则化项:系数平方和)
  - 作用：可以使得其中一些θ的都很小，都接近于0，削弱某个特征的影响
  - 优点：越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象
  - Ridge回归(常用)
- L1正则化(正则化项:系数绝对值的和)
  - 作用：可以使得其中一些θ的值直接为0，删除这个特征的影响
  - LASSO回归



```python
from sklearn.linear_model import Ridge,ElasticNet,Lasso
```

- 岭回归 RidgeRegression

  - API: 

    - sklearn.linear_model.Ridge(alpha=1.0)
    - 参数
      - alpha:  正则化力度

  - 正则化力度与系数之间的关系

    - 正则化力度越大, 模型系数越小
    - 正则化力度越小, 模型系数越大

  - 岭回归与线性回归的区别

    - 岭回归就是在线性回归上损失函数上添加L2的正则化项

    - 岭回归使用随机平均梯度下降法

    - 岭回归是一种线程回归

    - 具有l2正则化的线性回归，可以进行交叉验证

    - 总结: 岭回归是使用随机平均梯度下降法, 带有L2正则化的线性回归.

  - SGDRegressor 与 Ridge区别

    - SGDRegressor
      - 随机梯度下降法
    - Ridge
      - 随机平均梯度下降法
      - 使用L2正则化项.



#### 维灾难

随着维度的增加，分类器性能逐步上升，到达某点之后，其性能便逐渐下降







### 逻辑回归(经典二分类算法)

把线性回归的输出值映射为(0,1)范围,表示一个概率值,阈值默认为0.5,大于阈值为正例,反之为反例

注意点:阈值大小要结合实际情况







#### Sigmoid函数: 值-->概率




#### 损失

逻辑回归的损失，称之为对数似然损失，公式如下：

分开类别：


怎么理解单个的式子呢？这个要根据log的函数图像来理解



##### 损失函数



看到这个式子，其实跟我们讲的信息熵类似。



#### LogisticRegression API

```python
sklearn.linear_model.LogisticRegression(solver='liblinear', penalty=‘l2’, C = 1.0)
solver可选参数:{'liblinear', 'sag', 'saga','newton-cg', 'lbfgs'}，

默认: 'liblinear'；内部使用了坐标轴下降法迭代优化损失, 用于优化问题的算法。
对于小数据集来说，“liblinear”是个不错的选择，而“sag”和'saga'对于大型数据集会更快。
对于多类问题，只有'newton-cg'， 'sag'， 'saga'和'lbfgs'可以处理多项损失;“liblinear”仅限于“one-versus-rest”分类。

penalty：正则化的种类
C：正则化力度
默认将类别数量少的当做正例
LogisticRegression方法相当于 SGDClassifier(loss="log", penalty=" "),SGDClassifier实现了一个普通的随机梯度下降学习。而使用LogisticRegression(实现了SAG)
```



单词区分:     solver:解决者,slaver奴隶

####  肿瘤良性与恶性预测

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# 加载数据
names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                   'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                   'Normal Nucleoli', 'Mitoses', 'Class']

data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
                  names=names)
#data.head()

#处理数据
#处理缺失值
data.replace(to_replace='?',value=np.nan,inplace=True)
#删除缺失值所在行
data.dropna(inplace=True)#inplace=True 覆盖原数据
# np.any(data.isnull())
#选择特征值和目标值
x = data.iloc[:,1:-1]
y = data['Class']

#分割数据集
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.25,random_state=8)
#特征工程,标准化
transfer = StandardScaler()
x_train =transfer.fit_transform(x_train)
x_test =transfer.fit_transform(x_test)

#机器学习,模型训练:逻辑回归
estimator = LogisticRegression()
#模型训练
estimator.fit(x_train,y_train)

#模型评估
# y_pred = estimator.predict(x_test)
score =estimator.score(x_test,y_test)
print('准确率:',score)

#分类评估方式
from sklearn.metrics import classification_report

rs = classification_report(y_true =y_test,y_pred =y_pred,labels=[2,4],target_names=['良性','恶性'])
print(rs)
```

```python
准确率: 0.9707602339181286
              precision    recall  f1-score   support

          良性       0.97      0.98      0.98       104
          恶性       0.97      0.96      0.96        67

   micro avg       0.97      0.97      0.97       171
   macro avg       0.97      0.97      0.97       171
weighted avg       0.97      0.97      0.97       171
    
```

#### 逻辑回归和线性回归的计算区别在哪里？

线性回归 是以 高斯分布 为误差分析模型； 逻辑回归 采用的是 伯努利分布 分析误差。



逻辑回归的模型 是一个非线性模型，sigmoid函数，又称逻辑回归函数。但是它本质上又是一个线性回归模型，因为除去sigmoid映射函数关系，其他的步骤，算法都是线性回归的。可以说，逻辑回归，都是以线性回归为理论支持的。

只不过，线性模型，无法做到sigmoid的非线性形式，sigmoid可以轻松处理0/1分类问题。

另外它的推导含义：仍然与线性回归的最大似然估计推导相同，最大似然函数连续积（这里的分布，可以使伯努利分布，或泊松分布等其他分布形式），求导，得损失函数。



#### 逻辑回归的激活函数为什么用sigmoid？用ReLu可以吗？

使用Sigmoid作为激活函数:

Sigmod函数优点

输出范围有限，数据在传递的过程中不容易发散。

输出范围为(0,1)，所以可以用作输出层，输出表示概率。

抑制两头，对中间细微变化敏感，对分类有利。



不使用ReLu作为激活函数:



RELU特点：输入信号 <0 时，输出都是0，>0 的情况下，输出等于输入

ReLu的优点是梯度易于计算，而且梯度不会像sigmoid一样在边缘处梯度接近为0（梯度消失）。

ReLU 的缺点：

训练的时候很”脆弱”，很容易就”die”了

例如，一个非常大的梯度流过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了，那么这个神经元的梯度就永远都会是 0.

如果 learning rate 很大，那么很有可能网络中的 40% 的神经元都”dead”了。

Relu函数在神经元的值大于零的时候，Relu的梯度恒定为1，梯度在大于零的时候可以一直被传递。而且ReLU 得到的SGD的收敛速度会比 σσ、tanh 快很多。 

ReLU函数在训练的时候，一不小心有可能导致梯度为零。由于ReLU在x<0时梯度为0，这样就导致负的梯度在这个ReLU被置零，这个神经元有可能再也不会被任何数据激活，这个ReLU神经元坏死了，不再对任何数据有所响应。实际操作中，如果设置的learning rate 比较大，那么很有可能网络中的大量的神经元都坏死了。如果开始设置了一个合适的较小的learning rate，这个问题发生的情况其实也不会太频繁。



#### 精确率(Precision)与召回率(Recall)

- 准确率: 所有样本中预测正确的比例

$$
Accuracy = \frac{TP+TN}{TP+TN+FN+FP}
$$





- 精确率：预测结果为正例样本中真实为正例的比例（了解）

$$
Precision = \frac{TP}{TP+FP}
$$






- 召回率：真实为正例的样本中预测结果为正例的比例（查得全，对正样本的区分能力）

$$
Recall = \frac{TP}{TP + FN}
$$




TPR 即召回率 recall
FPR可以理解为错误率,此参数越低,损失越低


##### 案例.贷款申请最大化利润_Tyd



#### Roc曲线绘制

- ROC曲线和AUC指标:
  - 作用: 用于样本不均衡下模型评估
- ROC曲线:
  - 纵坐标: TPR: 真实类别为正例的样本中预测为正例的比例
  - 横坐标: FPR, 真实类别为反例的样本中预测为正例的比例
- AUC指标: 几何意义: ROC曲线下面面积(积分)
- AUC指标: 在[0.5, 1]之间, 越接近于1模型越接近与最佳模型, 越接近与0.5, 越是乱猜
- API:

```
sklearn.metrics.roc_auc_score(y_true, y_score)
```

- 参数
  - y_true： 真实的目标值, 要求0为反例, 1位正例
  - y_score：预测结果.

ROC曲线的绘制【**】

1. ROC绘制过程:
   1. 训练一个分类器模型
   2. 使用分类器模型算出测试样本的概率值
   3. 对概率值偶从大到小排序; 从第一个点开始, 计算TPR和FPR, 描点
   4. 使用线把所有点连接起来.

2. AUC指标: 几何意义ROC曲线下面的面积
   1. 当AUC指标为1的时候, 存在一个点, 可以完美的把数据分割开来
   2. AUC越接近与1分类的效果越好, 越接近0.5越是乱猜, 小于0.5乱猜都不如.






### 决策树

Decision tree

决策树思想的来源非常朴素，程序设计中的条件分支结构就是if-else结构，最早的决策树就是利用这类结构分割数据的一种分类学习方法

**决策树：是一种树形结构，其中每个内部节点表示一个属性上的判断，每个分支代表一个判断结果的输出，最后每个叶节点代表一种分类结果，本质是一颗由多个判断节点组成的树**。

重要的条件放在前面的结点



##### 信息熵

当数据量一致时,系统越有序,熵值越低;系统越混乱或者分散,熵值越高

决策树的过程就是不断熵减的过程,即越来越接近目标值




```python
 >>> -(2/5)*math.log(2/5,2)-(3/5)*math.log(3/5,2)
0.9709505944546686
```



#### 决策树便于可视化展示(绘图)



#### 信息增益率


#### 算法比较

| 算法 | 支持模型   | 树结构 | 特征选择         | 连续值处理 | 缺失值处理 | 剪枝   |
| ---- | ---------- | ------ | ---------------- | ---------- | ---------- | ------ |
| ID3  | 分类       | 多叉树 | 信息增益         | 不支持     | 不支持     | 不支持 |
| C4.5 | 分类       | 多叉树 | 信息增益比       | 支持       | 支持       | 支持   |
| CART | 分类，回归 | 二叉树 | 基尼系数，均方差 | 支持       | 支持       | 支持   |



##### 简述的ID3算法,C4.5算法,CART算法的优点和缺点

信息增益(ID.3)

作用: 用于衡量信息纯度的, 信息熵越小纯度越高.
公式: 
$$
Ent(D) = -(p_1log_2p_1 + p_2log_2p_2 +… + p_nlog_2p_n )
$$



信息增益(ID.3) 信息增益 = 整体信息上 - 按某个属性划分后的信息熵 信息增益 = entropy(前) - entropy(后)

信息增益率(C4.5决策树算法):

解决信息增益倾向于选择类别多属性进行划分. 
信息增益率 = 属性信息增益 / 属性分裂信息度量

基尼指数(CART决策树算法)
优点:

支持分类与回归模型,支持连续值处理与缺失值处理,缓解过拟合现象

缺点:
1）在做特征选择的时候都是选择最优的一个特征来做分类决策，但是大多数，分类决策不应该是由某一个特征决定的，而是应该由一组特征决定的。(多变量决策树)

2）如果样本发生一点点的改动，就会导致树结构的剧烈改变。(通过随机森林之类的方法解决)
3)有些比较复杂的关系，决策树很难学习，比如异或。(神经网络分类方法来解决)



#### 特征提取

- 字典特征提取(特征离散化)
- 文本特征提取
- 图像特征提取（深度学习将介绍）



##### 字典提取

```python
from sklearn.feature_extraction import DictVectorizer

datas=[{'city': '北京','temperature':100},
{'city': '上海','temperature':60},
{'city': '深圳','temperature':30}]

transfer =DictVectorizer(sparse=False)#是否返回稀疏矩阵
# transfer =DictVectorizer(sparse=True)
#字典特征提取,one-hot编码
new_datas =transfer.fit_transform(datas)
print(new_datas)
#获取每个特征的名称
print(transfer.get_feature_names())

-----output------
[[  0.   1.   0. 100.]
 [  1.   0.   0.  60.]
 [  0.   0.   1.  30.]]
['city=上海', 'city=北京', 'city=深圳', 'temperature']
```

对于特征当中存在类别信息的我们都会做one-hot编码处理

##### 文本提取

```python
from sklearn.feature_extraction.text import CountVectorizer #主要处理英文,对中文不友好

text=["life is short,i like python",
"life is too long,i dislike python",
     "life is short,i like Java"]

transfer =CountVectorizer()

#文本特征提取
new_data1 = transfer.fit_transform(text)
# print(new_data1)#稀疏矩阵
print(new_data1.toarray())
print(transfer.get_feature_names())
```





#### 文本分析

##### Tf-idf

- TF-IDF的主要思想是：如果**某个词或短语在一篇文章中出现的概率高，并且在其他文章中很少出现**，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
- **TF-IDF作用：用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。**





#####  公式

- 词频（term frequency，tf）指的是某一个给定的词语在该文件中出现的频率
- 逆向文档频率（inverse document frequency，idf）是一个词语普遍重要性的度量。某一特定词语的idf，可以**由总文件数目除以包含该词语之文件的数目，再将得到的商取以10为底的对数得到**

$$
tfidf_i,_j=tf_i,_j*idf_i
$$



最终得出结果可以理解为重要程度。

举例：
假如一篇文章的总词语数是100个，而词语"非常"出现了5次，那么"非常"一词在该文件中的词频就是5/100=0.05。
而计算文件频率（IDF）的方法是以文件集的文件总数，除以出现"非常"一词的文件数。
所以，如果"非常"一词在1,0000份文件出现过，而文件总数是10,000,000份的话，
其逆向文件频率就是lg（10,000,000 / 1,0000）=3。
最后"非常"对于这篇文档的tf-idf的分数为0.05 * 3=0.15



结巴分词器: jieba

Gensim     http://radimrehurek.com/gensim/

格式:list of list

语料库,停用词



##### 代码:中文分词
```python
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

def cut_word(text):
    """
    对中文进行分词
    "我爱北京天安门"————>"我 爱 北京 天安门"
    :param text:
    :return: text
    """
    # 用结巴对中文字符串进行分词
    #生成分词器
    text = " ".join(list(jieba.cut(text)))
    return text

def text_chinese_tfidf_demo():
    """
    对中文进行特征抽取
    :return: None
    """
    data = ["一种还是一种今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。",
            "我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去。",
            "如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。"]
    # 将原始数据转换成分好词的形式
    text_list = []
    for sent in data:
        text_list.append(cut_word(sent))
    print(text_list)

    # 1、实例化一个转换器类
    # transfer = CountVectorizer(sparse=False)
    transfer = TfidfVectorizer(stop_words=['一种', '不会', '不要'])
    # 2、调用fit_transform
    data = transfer.fit_transform(text_list)
    print("文本特征抽取的结果：\n", data.toarray())
    print("返回特征名字：\n", transfer.get_feature_names())
    return None
#调用
text_chinese_tfidf_demo()

------output------

['一种 还是 一种 今天 很 残酷 ， 明天 更 残酷 ， 后天 很 美好 ， 但 绝对 大部分 是 死 在 明天 晚上 ， 所以 每个 人 不要 放弃 今天 。', '我们 看到 的 从 很 远 星系 来 的 光是在 几百万年 之前 发出 的 ， 这样 当 我们 看到 宇宙 时 ， 我们 是 在 看 它 的 过去 。', '如果 只用 一种 方式 了解 某样 事物 ， 你 就 不会 真正 了解 它 。 了解 事物 真正 含义 的 秘密 取决于 如何 将 其 与 我们 所 了解 的 事物 相 联系 。']
文本特征抽取的结果：
 [[0.         0.         0.         0.43643578 0.         0.
  0.         0.         0.         0.21821789 0.         0.21821789
  0.         0.         0.         0.         0.21821789 0.21821789
  0.         0.43643578 0.         0.21821789 0.         0.43643578
  0.21821789 0.         0.         0.         0.21821789 0.21821789
  0.         0.         0.21821789 0.        ]
 [0.2410822  0.         0.         0.         0.2410822  0.2410822
  0.2410822  0.         0.         0.         0.         0.
  0.         0.         0.2410822  0.55004769 0.         0.
  0.         0.         0.2410822  0.         0.         0.
  0.         0.48216441 0.         0.         0.         0.
  0.         0.2410822  0.         0.2410822 ]
 [0.         0.644003   0.48300225 0.         0.         0.
  0.         0.16100075 0.16100075 0.         0.16100075 0.
  0.16100075 0.16100075 0.         0.12244522 0.         0.
  0.16100075 0.         0.         0.         0.16100075 0.
  0.         0.         0.3220015  0.16100075 0.         0.
  0.16100075 0.         0.         0.        ]]
返回特征名字：
 ['之前', '了解', '事物', '今天', '光是在', '几百万年', '发出', '取决于', '只用', '后天', '含义', '大部分', '如何', '如果', '宇宙', '我们', '所以', '放弃', '方式', '明天', '星系', '晚上', '某样', '残酷', '每个', '看到', '真正', '秘密', '绝对', '美好', '联系', '过去', '还是', '这样']
```

#### 决策树:Titanic生存预测

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report   


#1,加载数据
titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
#titan.head()
#2数据基本处理
#2.1选择特征值和目标值
x = titan[['pclass','age','sex']]
y= titan['survived']
#2.2缺失值处理,用均值填充
x['age'].fillna(x['age'].mean(),inplace=True)
# np.any(x.isnull())

#2.3数据分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 8)

#3特征工程(特征提取,字典特征提取)
# x_train.to_dict(orient='records')#把数据转换为字典格式
#创建字典特征提取器
transfer = DictVectorizer(sparse=False)
#字典特征提取
x_train =transfer.fit_transform(x_train.to_dict(orient='records'))
x_test =transfer.fit_transform(x_test.to_dict(orient='records'))
# print(x_test)
# print(transfer.get_feature_names())

#4机器学习,模型训练
estimator = DecisionTreeClassifier()
estimator.fit(x_train,y_train)


#5模型评估
y_pred = estimator.predict(x_test)
# print(y_pred)

#获取准确率
score =estimator.score(x_test,y_test)
print('准确率:',score)
#精确率precision,召回率recall
rs = classification_report(y_true =y_test,y_pred=y_pred)
print(rs)
#保存树的结构到dot文件
export_graphviz(estimator, out_file="./tree.dot", feature_names=['age', 'pclass=1st', 'pclass=2nd', 'pclass=3rd', '女性', '男性'])

-------output----------

准确率: 0.7918781725888325

              precision    recall  f1-score   support

           0       0.80      0.90      0.85       255
           1       0.77      0.59      0.67       139

   micro avg       0.79      0.79      0.79       394
   macro avg       0.78      0.75      0.76       394
weighted avg       0.79      0.79      0.78       394

```

#### 决策树可视化

##### 保存树的结构到dot文件

- sklearn.tree.export_graphviz() 该函数能够导出DOT格式
  - tree.export_graphviz(estimator,out_file='tree.dot’,feature_names=[‘’,’’])







### 集成算法

#### 集成学习算法简介

Ensemble learning

- Bagging
- Boosting
- Stacking

1. 什么是集成学习

   - 多个弱学习器组合在一起能力就变强.
   - 三个臭皮匠赛过诸葛亮

2. 机器学习两个核心任务:

   - 过拟合
     - 解决: 互相遏制变壮: bagging
   - 欠拟合
     - 解决: 弱弱组合变强: boosting




#### 随机森林



##### 用随机森林处理Titanic_analyse
```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report   
from sklearn.ensemble import RandomForestClassifier


#1,加载数据
titan = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
#titan.head()

#2数据基本处理
#2.1选择特征值和目标值
x = titan[['pclass','age','sex']]
y= titan['survived'] #获救结果
#2.2缺失值处理,用均值填充
x['age'].fillna(x['age'].mean(),inplace=True)
# np.any(x.isnull())

#2.3数据分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 8)

#3特征工程(特征提取,字典特征提取)
# x_train.to_dict(orient='records')#把数据转换为字典格式
#创建字典特征提取器
transfer = DictVectorizer(sparse=False)
#字典特征提取
x_train =transfer.fit_transform(x_train.to_dict(orient='records'))
x_test =transfer.fit_transform(x_test.to_dict(orient='records'))
# print(x_test)
# print(transfer.get_feature_names())

#4机器学习,模型训练
# estimator = DecisionTreeClassifier()
estimator = RandomForestClassifier()
#
param_grid = {'n_estimators':[10,50,100,200,500],'max_depth':[10,20,30,40,50]}


#交叉验证
estimator =GridSearchCV(estimator,param_grid=param_grid,cv=5)
estimator.fit(x_train,y_train)


#5模型评估
y_pred = estimator.predict(x_test)
# print(y_pred)

#获取准确率
score =estimator.score(x_test,y_test)
print('准确率:',score)
#精确率precision,召回率recall
rs = classification_report(y_true =y_test,y_pred=y_pred)
print(rs)

print('准确率:',score)
#风格搜索与交叉验证结果
print('交叉验证最好分数:',estimator.best_score_)
print('交叉验证最好的模型:',estimator.best_estimator_) #最佳模型与老师的不同
print('交叉验证的结果:\n',estimator.cv_results_)
```



##### 随机森林API

1. sklearn.ensemble.RandomForestClassifier(n_estimators=10, criterion=’gini’, max_depth=None, bootstrap=True, random_state=None, min_samples_split=2)
2. 参数
   1. n_estimators=10 : 弱决策树的数量
   2. max_depth=None: 决策树最大深度




#### Boosting模型 

典型代表:AdaBoost,Xgboost

串行训练器,防止欠拟合:

##### bagging集成与boosting集成的区别

区别一:数据方面

Bagging：对数据进行采样训练；

Boosting：根据前一轮学习结果调整数据的重要性。

区别二:投票方面

Bagging：所有学习器平权投票；

Boosting：对学习器进行加权投票。

区别三:学习顺序

Bagging的学习是并行的，每个学习器没有依赖关系；

Boosting学习是串行，学习有先后顺序。

区别四:主要作用

Bagging主要用于提高泛化性能（解决过拟合，也可以说降低方差）

Boosting主要用于提高训练精度 （解决欠拟合，也可以说降低偏差）



#### Xgboost

Xgboost中国人建立,有Python库

**XGBoost= 二阶泰勒展开+boosting+决策树+正则化**



在竞赛题中经常会用到XGBoost算法，用这个算法通常会使我们模型的准确率有一个较大的提升。既然它效果这么好，那么它从头到尾做了一件什么事呢？以及它是怎么样去做的呢？

我们先来直观的理解一下什么是XGBoost。XGBoost算法是和决策树算法联系到一起的。决策树算法在我的另一篇博客中讲过了.

##### 一、集成算法思想

在决策树中，我们知道一个样本往左边分或者往右边分，最终到达叶子结点，这样来进行一个分类任务。 其实也可以做回归任务。

![img](https://img-blog.csdn.net/20180713152759957?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h1YWNoYV9f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

看上面一个图例左边：有5个样本，现在想看下这5个人愿不愿意去玩游戏，这5个人现在都分到了叶子结点里面，对不同的叶子结点分配不同的权重项，正数代表这个人愿意去玩游戏，负数代表这个人不愿意去玩游戏。所以我们可以通过叶子结点和权值的结合，来综合的评判当前这个人到底是愿意还是不愿意去玩游戏。上面「tree1」那个小男孩它所处的叶子结点的权值是+2（可以理解为得分）。

用单个决策树好像效果一般来说不是太好，或者说可能会太绝对。通常我们会用一种集成的方法，就是一棵树效果可能不太好，用两棵树呢？

看图例右边的「tree2」，它和左边的不同在于它使用了另外的指标，出了年龄和性别，还可以考虑使用电脑频率这个划分属性。通过这两棵树共同帮我们决策当前这个人愿不愿意玩游戏，小男孩在「tree1」的权值是+2，在「tree2」的权值是+0.9， 所以小男孩最终的权值是+2.9（可以理解为得分是+2.9）。老爷爷最终的权值也是通过一样的过程得到的。

所以说，我们通常在做分类或者回归任务的时候，需要想一想一旦选择用一个分类器可能表达效果并不是很好，那么就要考虑用这样一个集成的思想。上面的图例只是举了两个分类器，其实还可以有更多更复杂的弱分类器，一起组合成一个强分类器。

 

##### 二、XGBoost基本思想

**XGBoost= 二阶泰勒展开+boosting+决策树+正则化**

XGBoost的集成表示是什么？怎么预测？求最优解的目标是什么？看下图的说明你就能一目了然。

![img](https://img-blog.csdn.net/20180713152916377?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h1YWNoYV9f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

在XGBoost里，每棵树是一个一个往里面加的，每加一个都是希望效果能够提升，下图就是XGBoost这个集成的表示（核心）。

![img](https://img-blog.csdn.net/20180713152950675?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h1YWNoYV9f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

一开始树是0，然后往里面加树，相当于多了一个函数，再加第二棵树，相当于又多了一个函数...等等，这里需要保证加入新的函数能够提升整体对表达效果。提升表达效果的意思就是说加上新的树之后，目标函数（就是损失）的值会下降。

如果叶子结点的个数太多，那么过拟合的风险会越大，所以这里要限制叶子结点的个数，所以在原来目标函数里要加上一个惩罚项「omega(ft)」。

![img](https://img-blog.csdn.net/20180713155211262?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h1YWNoYV9f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 

这里举个简单的例子看看惩罚项「omega(ft)」是如何计算的：

![img](https://img-blog.csdn.net/2018071315553980?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h1YWNoYV9f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

一共3个叶子结点，权重分别是2，0.1，-1，带入「omega(ft)」中就得到上面图例的式子，惩罚力度和「lambda」的值人为给定。

XGBoost算法完整的目标函数见下面这个公式，它由自身的损失函数和正则化惩罚项「omega(ft)」相加而成。

![img](https://img-blog.csdn.net/20180713160355939?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h1YWNoYV9f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

关于目标函数的推导本文章不作详细介绍。过程就是：给目标函数对权重求偏导，得到一个能够使目标函数最小的权重，把这个权重代回到目标函数中，这个回代结果就是求解后的最小目标函数值，如下：




其中第三个式子中的一阶导二阶导的梯度数据都是可以算出来的，只要指定了主函数中的两个参数，这就是一个确定的值。下面给出一个直观的例子来看下这个过程。

![img](https://img-blog.csdn.net/20180713163300615?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h1YWNoYV9f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

（这里多说一句：Obj代表了当我们指定一个树的结构的时候，在目标上最多会减少多少，我们可以把它叫做结构分数，这个分数越小越好）

对于每次扩展，我们依旧要枚举所有可能的方案。对于某个特定的分割，我们要计算出这个分割的左子树的导数和和右子数导数和之和（就是下图中的第一个红色方框），然后和划分前的进行比较（基于损失，看分割后的损失和分割前的损失有没有发生变化，变化了多少）。遍历所有分割，选择变化最大的作为最合适的分割。

![img](https://img-blog.csdn.net/20180713170150129?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2h1YWNoYV9f/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

 



##### 用python实现XGBoost算法

pima-indians-diabetes.csv文件中包括了8列数值型自变量，和第9列0-1的二分类因变量，导入到python中用XGBoost算法做探索性尝试，得到预测数据的准确率为77.95%。

```python
import xgboost
from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 载入数据集
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=",")

# split data into X and y
X = dataset[:,0:8]
Y = dataset[:,8]

#把数据集拆分成训练集和测试集
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

#拟合XGBoost模型
model = XGBClassifier()
model.fit(X_train, y_train)

#对测试集做预测
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

#评估预测结果
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

结果输出：
Accuracy: 77.95%
```

#### GBDT梯度提升决策树

(GBDT Gradient Boosting Decision Tree) 

##### GBDT主要执行思想

1.使用梯度下降法优化代价函数；

2.使用一层决策树作为弱学习器，负梯度作为目标值；

3.利用boosting思想进行集成。











### 集成学习算法代码

#### 决策树, 随机森林, KNN算法

分别使用决策树, 随机森林, KNN算法, 并使用网格搜索和交叉验证对参数进行调优; 得到一个最佳的算法模型.

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report   
```

#### 决策树

```python
#1,加载数据
data = pd.read_csv("./UCI_Credit_Card.csv")
data.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>120000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272.0</td>
      <td>3455.0</td>
      <td>3261.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>90000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>50000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>50000.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>

</div>



```python
#看是否有空值
print(np.any(data.isnull()))
```

```
False
```



```python
x = data[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
           'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'
           ,'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]

#x = data.iloc[:, 1:-1].values

y= data['default.payment.next.month']


#2.3数据分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 8)

#3特征工程(特征提取,字典特征提取)
# x_train.to_dict(orient='records')#把数据转换为字典格式
#创建字典特征提取器
transfer = DictVectorizer(sparse=False)
#字典特征提取
x_train =transfer.fit_transform(x_train.to_dict(orient='records'))
x_test =transfer.fit_transform(x_test.to_dict(orient='records'))
# print(x_test)
# print(transfer.get_feature_names())

#4机器学习,模型训练
estimator = DecisionTreeClassifier()
estimator.fit(x_train,y_train)


```



```
DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
            max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best')
```



```python
#5模型评估
y_pred = estimator.predict(x_test)
# print(y_pred)

#获取准确率
score =estimator.score(x_test,y_test)
print('准确率:',score)
#精确率precision,召回率recall
rs = classification_report(y_true =y_test,y_pred=y_pred)
print(rs)
```

```
准确率: 0.7288888888888889
              precision    recall  f1-score   support

           0       0.83      0.82      0.82      6995
           1       0.40      0.42      0.41      2005

   micro avg       0.73      0.73      0.73      9000
   macro avg       0.61      0.62      0.62      9000
weighted avg       0.73      0.73      0.73      9000
```

​    

#### 随机森林

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
```

```python
x = data[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
           'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'
           ,'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]
y= data['default.payment.next.month']


#2.3数据分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 8)

#3特征工程(特征提取,字典特征提取)
# x_train.to_dict(orient='records')#把数据转换为字典格式
#创建字典特征提取器
transfer = DictVectorizer(sparse=False)
#字典特征提取
x_train =transfer.fit_transform(x_train.to_dict(orient='records'))
x_test =transfer.fit_transform(x_test.to_dict(orient='records'))
# print(x_test)
# print(transfer.get_feature_names())

#4机器学习,模型训练
estimator = RandomForestClassifier()
# param_grid = {'n_estimators':[10,50,100,200,500],'max_depth':[10,20,30,40,50]}


# #交叉验证
# estimator =GridSearchCV(estimator,param_grid=param_grid,cv=5)
estimator.fit(x_train,y_train)


#5模型评估
y_pred = estimator.predict(x_test)
# print(y_pred)

#获取准确率
score =estimator.score(x_test,y_test)
print('准确率:',score)
#精确率precision,召回率recall
rs = classification_report(y_true =y_test,y_pred=y_pred)
print(rs)
```

```
c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\sklearn\ensemble\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
  "10 in version 0.20 to 100 in 0.22.", FutureWarning)
```

```
准确率: 0.8071111111111111
              precision    recall  f1-score   support

           0       0.83      0.95      0.88      6995
           1       0.63      0.32      0.43      2005

   micro avg       0.81      0.81      0.81      9000
   macro avg       0.73      0.63      0.66      9000
weighted avg       0.79      0.81      0.78      9000
```

​    

#### 随机森林,交叉验证

```python
#4机器学习,模型训练
estimator = RandomForestClassifier()
param_grid = {'n_estimators':[10,50,100,200,500],'max_depth':[10,20,30,40,50]}


#交叉验证
estimator =GridSearchCV(estimator,param_grid=param_grid,cv=5)
estimator.fit(x_train,y_train)


#5模型评估
y_pred = estimator.predict(x_test)
# print(y_pred)

#获取准确率
score =estimator.score(x_test,y_test)
print('准确率:',score)
#精确率precision,召回率recall
rs = classification_report(y_true =y_test,y_pred=y_pred)
print(rs)
```

```python

```

#### KNN

```python
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
```

```python
#1,加载数据
data = pd.read_csv("./UCI_Credit_Card.csv")
data.head()
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

```
.dataframe tbody tr th {
    vertical-align: top;
}

.dataframe thead th {
    text-align: right;
}
```

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>LIMIT_BAL</th>
      <th>SEX</th>
      <th>EDUCATION</th>
      <th>MARRIAGE</th>
      <th>AGE</th>
      <th>PAY_0</th>
      <th>PAY_2</th>
      <th>PAY_3</th>
      <th>PAY_4</th>
      <th>...</th>
      <th>BILL_AMT4</th>
      <th>BILL_AMT5</th>
      <th>BILL_AMT6</th>
      <th>PAY_AMT1</th>
      <th>PAY_AMT2</th>
      <th>PAY_AMT3</th>
      <th>PAY_AMT4</th>
      <th>PAY_AMT5</th>
      <th>PAY_AMT6</th>
      <th>default.payment.next.month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>20000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>24</td>
      <td>2</td>
      <td>2</td>
      <td>-1</td>
      <td>-1</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>689.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>120000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>26</td>
      <td>-1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>3272.0</td>
      <td>3455.0</td>
      <td>3261.0</td>
      <td>0.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>0.0</td>
      <td>2000.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>90000.0</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>34</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>14331.0</td>
      <td>14948.0</td>
      <td>15549.0</td>
      <td>1518.0</td>
      <td>1500.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>1000.0</td>
      <td>5000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>50000.0</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>28314.0</td>
      <td>28959.0</td>
      <td>29547.0</td>
      <td>2000.0</td>
      <td>2019.0</td>
      <td>1200.0</td>
      <td>1100.0</td>
      <td>1069.0</td>
      <td>1000.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>50000.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>57</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>0</td>
      <td>...</td>
      <td>20940.0</td>
      <td>19146.0</td>
      <td>19131.0</td>
      <td>2000.0</td>
      <td>36681.0</td>
      <td>10000.0</td>
      <td>9000.0</td>
      <td>689.0</td>
      <td>679.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 25 columns</p>

</div>



```python
x = data[['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
           'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'
           ,'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']]
y= data['default.payment.next.month']


#2.3数据分割
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size =0.3,random_state = 8)

#机器学习 KNN+GridSearchCV
estimator =KNeighborsClassifier()

#学习:训练集特征值,训练集目标值
estimator.fit(x_train,y_train)
#模型评估
#使用模型对测试集进行预测,参数为测试集的特征值,返回为预测目标值
y_pre =estimator.predict(x_test)


#准确率
score =estimator.score(x_test,y_test)
print('准确率:',score)

```

```
准确率: 0.7517777777777778
```

#### KNN + 交叉验证

```python
#网格搜索和交叉验证进行参数调优
#超参数数字典
param_grid ={'n_neighbors':[3,5,7,9]}
estimator=GridSearchCV(estimator,param_grid=param_grid,cv=5)

#学习:训练集特征值,训练集目标值
estimator.fit(x_train,y_train)
#模型评估
#使用模型对测试集进行预测,参数为测试集的特征值,返回为预测目标值
y_pre =estimator.predict(x_test)


#准确率
score =estimator.score(x_test,y_test)
print('准确率:',score)
#风格搜索与交叉验证结果
print('交叉验证最好分数:',estimator.best_score_)
print('交叉验证最好的模型:',estimator.best_estimator_) #最佳模型与老师的不同
print('交叉验证的结果:\n',estimator.cv_results_)

```

#### key_teacher(封装)

```python
# -*- coding: utf-8 -*-
# 信用卡违约率分析
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


# 数据加载
data = data = pd.read_csv('./UCI_Credit_Card.csv')
# 数据探索

# 选择有效的特征值
# 特征值, 去掉第一个ID和最后一个类别
# 注意在使用Pipeline封装后的评估器, 数据中不能有列名, 此处需要DataFrame中的values.
x = data.iloc[:, 1:-1].values
# 目标值
y = data['default.payment.next.month'].values

# 30% 作为测试集，其余作为训练集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

#print(x_train)

# 构造各种分类器
classifiers = [
    # 决策树
    DecisionTreeClassifier(random_state=1, criterion='gini'),
    # 随机森林
    RandomForestClassifier(random_state=1, criterion='gini'),
    # K近邻
    KNeighborsClassifier(metric='minkowski'),
]
# 分类器名称
classifier_names = [
    'dt', # 决策树
    'rf', # 随机森林
    'knn', # K近邻
]

# 使用网格搜索时, 分类器的超参数
classifier_param_grid = [
    # Pipeline: 封装后的评估器会根据模型名称, 获取对应的超参数
    # 格式要求: 模型名__超参数名称
    {'dt__max_depth': [5, 7, 9]},    # 决策树超参数
    {'rf__n_estimators': [3, 5, 6]}, # 随机森林超参数
    {'knn__n_neighbors': [4, 6, 8]}, # knn超参数
]


# 对具体的分类器进行 GridSearchCV 参数调优
def GridSearchCV_work(pipeline, x_train, y_train, x_test, y_test, param_grid, score='accuracy'):
    rs = {}
    gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score, cv=5)
    # 寻找最优的参数 和最优的准确率分数
    search = gridsearch.fit(x_train, y_train)
    print("GridSearch 最优参数：", search.best_params_)
    print("GridSearch 最优分数： %0.4lf" % search.best_score_)
    y_predict = gridsearch.predict(x_test)
    print(" 准确率 %0.4lf" % accuracy_score(y_test, y_predict))
    rs['y_predict'] = y_predict
    rs['accuracy_score'] = accuracy_score(y_test, y_predict)
    return rs

# 遍历获取模型, 模型名称, 模型超参数
for model, model_name, model_param_grid in zip(classifiers, classifier_names, classifier_param_grid):

    # 使用管道将标准化和模型封装成为为一个Pipeline评估器
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        (model_name, model)
    ])
    # print(model_param_grid)
    # 使用网格搜索与交叉验证选择模型
    result = GridSearchCV_work(pipeline, x_train, y_train, x_test, y_test, model_param_grid, score='accuracy')
```

```
[[2.0000e+04 1.0000e+00 1.0000e+00 ... 0.0000e+00 0.0000e+00 0.0000e+00]
 [1.2000e+05 1.0000e+00 2.0000e+00 ... 5.0000e+03 3.0000e+03 3.0000e+03]
 [7.0000e+04 2.0000e+00 2.0000e+00 ... 7.5500e+02 1.0290e+03 5.3030e+03]
 ...
 [5.0000e+04 1.0000e+00 2.0000e+00 ... 7.0500e+02 8.1100e+02 8.7400e+02]
 [8.0000e+04 1.0000e+00 1.0000e+00 ... 1.3780e+03 1.9942e+04 2.4180e+03]
 [3.0000e+05 1.0000e+00 2.0000e+00 ... 6.6000e+01 1.4062e+04 3.0810e+03]]
GridSearch 最优参数： {'dt__max_depth': 5}
GridSearch 最优分数： 0.8202
 准确率 0.8198
GridSearch 最优参数： {'rf__n_estimators': 6}
GridSearch 最优分数： 0.7974
 准确率 0.7988
```



```
---------------------------------------------------------------------------

KeyboardInterrupt                         Traceback (most recent call last)

<ipython-input-24-5a6ede94da00> in <module>
     77     # print(model_param_grid)
     78     # 使用网格搜索与交叉验证选择模型
---> 79     result = GridSearchCV_work(pipeline, x_train, y_train, x_test, y_test, model_param_grid, score='accuracy')
```

```
<ipython-input-24-5a6ede94da00> in GridSearchCV_work(pipeline, x_train, y_train, x_test, y_test, param_grid, score)
     58     gridsearch = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring=score, cv=5)
     59     # 寻找最优的参数 和最优的准确率分数
---> 60     search = gridsearch.fit(x_train, y_train)
     61     print("GridSearch 最优参数：", search.best_params_)
     62     print("GridSearch 最优分数： %0.4lf" % search.best_score_)
```













### Stacking模型

堆叠,聚合多个分类或回归模型

堆叠在一直确实能使得准确率提升,但是速度是个问题

集成算法是竞赛与论文神器,当我们更关注于结果时不妨来试试!

在一定程度上可以防止过拟合







### 支持向量机SVM

之前很火,后面遇到对手:神经网络



#### 核心(优化目标)

##### 找到一条线(w,b),使得离该线最近的点(雷区)能够最远








### 贝叶斯









### 聚类算法(无监督) 

无监督问题：我们手里没有标签了

聚类：相似的东西分到一组, eg.按照不同标准对事物进行分类

难点：如何评估，如何调参

#### 应用:

用户画像，广告推荐，Data Segmentation，搜索引擎的流量推荐，恶意流量识别

种类:粗聚类,细聚类



聚类与分类区别:聚类无目标值而分类有



#### K-means算法

##### 基本概念

要得到簇的个数，需要指定K值

质心：均值，即向量各维取平均即可

距离的度量：常用欧几里得距离和余弦相似度（先标准化）



优势:简单，快速，适合常规数据集

劣势:K值难确定,复杂度与样本呈线性关系,很难发现任意形状的簇



##### api介绍

- sklearn.cluster.KMeans(n_clusters=8)
  - 参数:
    - n_clusters:开始的聚类中心数量
      - 整型，缺省值=8，生成的聚类数，即产生的质心（centroids）数。
  - 方法:
    - estimator.fit(x)
    - estimator.predict(x)
    - estimator.fit_predict(x)
      - 计算聚类中心并预测每个样本属于哪个类别,相当于先调用fit(x),然后再调用predict(x)



##### 模型评估

**1.** **肘部法**

 下降率突然变缓时即认为是最佳的k值

**2.** **SC系数**

 取值为[-1, 1]，其值越大越好

**3.** **CH系数**

 分数s高则聚类效果越好

```python
from sklearn.metrics import mean_squared_error,mean_absolute_error    
#mean_squared_error均方误差,mean_absolute_error平均均方误差

mae = mean_absolute_error(y_test,y_pre)
print('平均绝对误差:',mae)
```



##### 算法优化

| **优化方法**       | **思路**                                               |
| ------------------ | ------------------------------------------------------ |
| Canopy+kmeans      | Canopy粗聚类配合kmeans,同心圆                          |
| kmeans++           | 距离越远越容易成为新的质心                             |
| 二分k-means        | 拆除SSE最大的簇,                                       |
| k-medoids          | 和kmeans选取中心点的方式不同                           |
| kernel kmeans      | 映射到高维空间                                         |
| ISODATA            | 动态聚类,合并:类别间距离小,分裂:类别内部方差大         |
| Mini-batch K-Means | 大数据集分批聚类,随机不放回,对小批量进行KMeans进行聚类 |



#### DBSCAN算法

核心对象：若某个点的密度达到算法设定的阈值则其为核心点。
（即 r 邻域内点的数量不小于 minPts）



### 特征工程



#### 降维

降维,eg.三维的地球,降维至平面地图

方式

- **特征选择**,从原有特征中找出主要特征
- **主成分分析（可以理解一种特征提取的方式）**



sklearn.feature_extraction  # 特征提取
sklearn.feature_selection   #特征选择



### 相关系数

#### 皮尔逊相关系数

使用 Scipy库

```python
from scipy.stats import pearsonr

x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]

pearsonr(x1, x2)
-------
(0.9941983762371883, 4.9220899554573455e-09)
```

##### 特点

**相关系数的值介于–1与+1之间，即–1≤ r ≤+1**。其性质如下：

- **当r>0时，表示两变量正相关，r<0时，两变量为负相关**
- 当|r|=1时，表示两变量为完全相关，当r=0时，表示两变量间无相关关系
- **当0<|r|<1时，表示两变量存在一定程度的相关。且|r|越接近1，两变量间线性关系越密切；|r|越接近于0，表示两变量的线性相关越弱**
- **一般可按三级划分：|r|<0.4为低度相关；0.4≤|r|<0.7为显著性相关；0.7≤|r|<1为高度线性相关**

pandas中corr方法可直接用于计算皮尔逊相关系数





#### 斯皮尔曼相关系数

```python
from scipy.stats import spearmanr

x1 = [12.5, 15.3, 23.2, 26.4, 33.5, 34.4, 39.4, 45.2, 55.4, 60.9]
x2 = [21.2, 23.9, 32.9, 34.1, 42.5, 43.2, 49.0, 52.8, 59.4, 63.5]

spearmanr(x1, x2)

-----
SpearmanrResult(correlation=0.9999999999999999, pvalue=6.646897422032013e-64)
```

**特点**

- 斯皮尔曼相关系数表明 X (自变量) 和 Y (因变量)的相关方向。 如果当X增加时， Y 趋向于增加, 斯皮尔曼相关系数则为正
- 与之前的皮尔逊相关系数大小性质一样，取值 [-1, 1]之间



#### 主成分分析

```python
from sklearn.decomposition import PCA
data = [[2,8,4,5], [6,3,0,8], [5,4,9,1]]

#创建主成分分析对象
#如果是小数,表示保留的信息的百分比
transfer =PCA(n_components=0.9)
#使用主成分分析进行特征降维
data1=transfer.fit_transform(data)
print("保留90%的信息，降维结果为：\n", data1)

-----
保留90%的信息，降维结果为：
 [[ 1.28620952e-15  3.82970843e+00]
 [ 5.74456265e+00 -1.91485422e+00]
 [-5.74456265e+00 -1.91485422e+00]]

```



#### 案例：探究用户对物品类别的喜好细分降维

```python
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import calinski_harabaz_score,silhouette_score

#1,加载数据
orders_data = pd.read_csv('../data_instacart_day9/orders.csv')
order_products__prior =pd.read_csv('../data_instacart_day9/order_products__prior.csv')
products = pd.read_csv('../data_instacart_day9/products.csv')
aisles = pd.read_csv('../data_instacart_day9/aisles.csv')

# 2数据基本处理
#数据合并
table1 = pd.merge(orders_data,order_products__prior,on='order_id')
table2 = pd.merge(table1,products,on='product_id')
table3 =pd.merge(table2,aisles,on='aisle_id')
#交叉表统计
print(table3.shape)
table = pd.crosstab(index=table3['user_id'],columns=table3['aisle'])
print(table.shape)

#3,特征工程(特征降维,主成分分析)
transfer =PCA(n_components=0.9)
datas =transfer.fit_transform(table)


#4,机器学习
estimator =KMeans(n_clusters=8,random_state=22)
y_pred=estimator.fit_predict(datas)

#5,模型评估
#计算所有样本的平均轮廓系数
silhouette_score(datas,y_pred)

--------
(32434489, 14)
(206209, 134)
0.3348187287765577
```



使用每个样本`a`的平均簇内距离（）和平均最近簇距离（`b`）来计算剪影系数。样本的Silhouette系数是。澄清一下，是样本与样本不属于的最近聚类之间的距离。请注意，仅在标签数为2 <= n_labels <= n_samples - 1时才定义Silhouette Coefficient。`(b - a) / max(a, b)``b`

此函数返回所有样本的平均Silhouette系数。要获取每个样品的值，请使用[`silhouette_samples`](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html#sklearn.metrics.silhouette_samples)。

最佳值为1，最差值为-1。接近0的值表示重叠的簇。负值通常表示已将样本分配给错误的群集，因为不同的群集更相似。





### 算法选择



###  模型评估

#### 分类模型评估

- 准确率
  - 预测正确的数占样本总数的比例。
- 精确率
  - 正确预测为正占**全部预测为正**的比例
- 召回率
  - 正确预测为正占**全部正样本**的比例
- F1-score
  - 主要用于评估模型的稳健性
- AUC指标
  - 主要用于评估样本不均衡的情况

##### 精确率(Precision)与召回率(Recall)

- 准确率: 所有样本中预测正确的比例

$$
Accuracy = \frac{TP+TN}{TP+TN+FN+FP}
$$





- 精确率：预测结果为正例样本中真实为正例的比例（了解）

$$
Precision = \frac{TP}{TP+FP}
$$





- 召回率：真实为正例的样本中预测结果为正例的比例（查得全，对正样本的区分能力）

$$
Recall = \frac{TP}{TP + FN}
$$




TPR 即召回率 recall
FPR可以理解为错误率,此参数越低,损失越低






#### 回归模型评估

- ##### 均方根误差（Root Mean Squared Error，RMSE）

  - RMSE是一个衡量回归模型误差率的常用公式。 然而，它仅能比较误差是相同单位的模型。


- ##### 相对平方误差（Relative Squared Error，RSE）

  - 与RMSE不同，RSE可以比较误差是不同单位的模型。





- ##### 平均绝对误差（Mean Absolute Error，MAE)

  - MAE与原始数据单位相同， 它仅能比较误差是相同单位的模型。量级近似与RMSE，但是误差值相对小一些。



- ##### 相对绝对误差（Relative Absolute Error，RAE)

  - 与RSE不同，RAE可以比较误差是不同单位的模型。



- ##### 决定系数 (Coefficient of Determination)

  - 决定系数 (**R2**)回归模型汇总了回归模型的解释度，由平方和术语计算而得。



  - R2描述了回归模型所解释的因变量方差在总方差中的比例。R2很大，即自变量和因变量之间存在线性关系，如果回归模型是“完美的”，SSE为零，则R2为1。R2小，则自变量和因变量之间存在线性关系的证据不令人信服。如果回归模型完全失败，SSE等于SST，没有方差可被回归解释，则R2为零。

    - 注:

      - SSE: The sum of squares due to error(误差平方和)



      - SST: Total sum of squares(原始数据和均值之差的平方和)



  - 当R2越接近1时，表示相关的`方程式`参考价值越高；相反，越接近0时，表示参考价值越低



### 统计函数

![img](D:/002--------------/create6@126.com/9ae0fd5564474e1a83ff9afcde2d35ca/clipboard.png)

#### 常用统计公式

![img](D:/002--------------/create6@126.com/d19f92a8a3ca4474af90e639db03b672/clipboard.png)



#### 数学实际含义

- 方差（Variance）：用来度量随机变量和其数学期望（即均值）之间的偏离程度。
- 标准差：方差开根号。
- 协方差：衡量两个变量之间的变化方向关系。
- 方差、标准差、和协方差之间的联系与区别：

方差和标准差都是对一组（一维）数据进行统计的，反映的是一维数组的离散程度；而协方差是对2维数据进行的，反映的是2组数据之间的相关性。

标准差和均值的量纲（单位）是一致的，在描述一个波动范围时标准差比方差更方便。

方差可以看成是协方差的一种特殊情况，即2组数据完全相同。

协方差只表示线性相关的方向，取值正无穷到负无穷。

协方差只是说明了线性相关的方向，说不能说明线性相关的程度，若衡量相关程度，则使用相关系数



相关英文单词:

actual  target 真实值

predicted target  预测值



#### 均方根误差:



#### 方差、标准差、协方差区别

1、定义不同

统计中的方差（样本方差）是每个样本值与全体样本值的平均数之差的平方值的平均数；

标准差是总体各单位标准值与其平均数离差平方的算术平均数的平方根；

协方差表示的是两个变量的总体的误差，这与只表示一个变量误差的方差不同。

2、计算方法不同

方差的计算公式为：





![img](https://gss0.baidu.com/-vo3dSag_xI4khGko9WTAnF6hhy/zhidao/wh%3D600%2C800/sign=ab39b415bd1c8701d6e3bae0174fb217/d53f8794a4c27d1e5751313515d5ad6edcc438cf.jpg)



式中的s²表示方差，x1、x2、x3、.......、xn表示样本中的各个数据，M表示样本平均数；

标准差=方差的算术平方根=s=sqrt(((x1-x)^2 +(x2-x)^2 +......(xn-x)^2)/n)；

协方差计算公式为：Cov(X,Y)=E[XY]-E[X]E[Y]，其中E[X]与E[Y]是两个实随机变量X与Y的期望值。





![img](https://gss0.baidu.com/9fo3dSag_xI4khGko9WTAnF6hhy/zhidao/wh%3D600%2C800/sign=c0f34d32d9ca7bcb7d2ecf298e39475b/42a98226cffc1e17387ff8c04490f603728de97b.jpg)



3、意义不同

方差和标准差都是对一组(一维)数据进行统计的，反映的是一维数组的离散程度；

而协方差是对2组数据进行统计的，反映的是2组数据之间的相关性。



##### 协方差





将第一个公式中括号内的完全平方打开得到

DX=E(X^2-2XEX+(EX)^2)

=E(X^2)-E(2XEX)+(EX)^2

=E(X^2)-2(EX)^2+(EX)^2

=E(X^2)-(EX)^2

若随机变量X的分布函数F(x)可表示成一个非负可积函数f(x)的积分，则称X为连续性随机变量，f(x)称为X的概率密度函数（分布密度函数）。

**协方差矩阵中主对角线为方差**



### 线性代数

#### 矩阵

[1,2,3,4] 默认表示列向量,如何表示行向量?

a=np.array([1,2,3])

a.shape

(3,)

b=np.array([[1,2,3]])

b.shape

(1, 3)



np.array 切片:






### 深度学习


当数据量变大时,深度学习的效果更好



#### 常规思路

##### 1.收集数据并给定标签

##### 2.训练一个分类器

##### 3.测试评估







### 神经网络

#### 结构



激活函数(非线性)

![1565945624731](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565945624731.png)

##### 神经网络首选ReLU激活函数





Sigmoid作为激活函数:

Sigmod函数优点

输出范围有限，数据在传递的过程中不容易发散。

输出范围为(0,1)，所以可以用作输出层，输出表示概率。

抑制两头，对中间细微变化敏感，对分类有利。



ReLu作为激活函数:

RELU特点：输入信号 <0 时，输出都是0，>0 的情况下，输出等于输入

ReLu的优点是梯度易于计算，而且梯度不会像sigmoid一样在边缘处梯度接近为0（梯度消失）。

ReLU 的缺点：

训练的时候很”脆弱”，很容易就”die”了

例如，一个非常大的梯度流过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了，那么这个神经元的梯度就永远都会是 0.

如果 learning rate 很大，那么很有可能网络中的 40% 的神经元都”dead”了。

Relu函数在神经元的值大于零的时候，Relu的梯度恒定为1，梯度在大于零的时候可以一直被传递。而且ReLU 得到的SGD的收敛速度会比 σσ、tanh 快很多。 

ReLU函数在训练的时候，一不小心有可能导致梯度为零。由于ReLU在x<0时梯度为0，这样就导致负的梯度在这个ReLU被置零，这个神经元有可能再也不会被任何数据激活，这个ReLU神经元坏死了，不再对任何数据有所响应。实际操作中，如果设置的learning rate 比较大，那么很有可能网络中的大量的神经元都坏死了。如果开始设置了一个合适的较小的learning rate，这个问题发生的情况其实也不会太频繁。





## 推荐系统简介


浏览-搜索-推荐-广告

搜索:主动,需要提供关键词

推荐:被动,不需要提供关键词,根据客户喜好推荐,排序,锦上添花,解决信息过载,用户没有明确需求(eg.商场导购)



eg.推荐电影,推荐汽车,推荐房产,推荐股票,推荐贷款产品



#### 推荐系统 V.S. 搜索引擎

|          | 搜索     | 推荐     |
| -------- | -------- | -------- |
| 行为方式 | 主动     | 被动     |
| 意图     | 明确     | 模糊     |
| 个性化   | 弱       | 强       |
| 流量分布 | 马太效应 | 长尾效应 |
| 目标     | 快速满足 | 持续服务 |
| 评估指标 | 简明     | 复杂     |





##### 推荐系统原理

社会化推荐:朋友咨询,让好友推荐

基于内容的推荐:通过文本找出相似的内容

基于流行度推荐,排行榜

基于协同过滤的推荐:找到兴趣小组,找有相似兴趣的用户或小组,看看他们最近的行为(eg.看的电影),物以类聚,人以群分



##### 作用:

高效连接用户和物品,

提高用户停留时间和用户活跃程序,

有效的帮助产品实现其商业价值(会有业绩考核,转化率指标)

应用场景:头条,淘宝,垂直领域领头企业





#### 推荐系统和Web项目的区别

- 通过信息过滤实现目标提升 V.S. 稳定的信息流通系统

- - web项目: 处理复杂业务逻辑，处理高并发，为用户构建一个稳定的信息流通服务
  - 推荐系统: 追求指标增长, 留存率/阅读时间/GMV (Gross Merchandise Volume电商网站成交金额)/视频网站VV (Video View)

- 确定 V.S. 不确定思维

- - web项目: 对结果有确定预期
  - 推荐系统: 结果是概率问题,**不确定性**

概率思想,统计思想





#### 推荐系统要素

- UI 和 UE(前端界面)
- 数据 (Lambda架构)
- 业务知识
- 算法



1EB = 1024 PB

1PB = 1024 TB




#### 大数据Lambda架构

- Lambda架构是由实时大数据处理框架Storm的作者Nathan Marz提出的一个实时大数据处理框架。
- Lambda架构的将离线计算和实时计算整合，设计出一个能满足实时大数据系统关键特性的架构，包括有：高容错、低延时和可扩展等。
- 离线计算(计算量大,耗时长,Hadoop,模型训练)



- 离线计算
  - 慢 处理的数据量比较大
  - hadoop
    - hdfs 数据存储
    - mapreduce
  - 模型训练
  - 数据处理
- 实时计算
  - 低延迟
  - 处理的数据量会小
  - 训练好的模型加载 实时的排序
  - 用户的实时特征变化 捕捉 更新数据
- Lambda架构解决的问题
  - 离线计算和实时计算协同工作的问题
  - 高容错、低延时和可扩展 的服务
- 推荐系统当中lambda架构的作用
  - 离线计算训练模型
  - 实时计算提供推荐服务
  - 实时计算还可以实时调整用户的特征和用户感兴趣的商品结果



#### 分层架构

- 批处理层
  - 数据不可变, 可进行任何计算, 可水平扩展
  - 高延迟 几分钟~几小时(计算量和数据量不同)
  - 日志收集： Flume
  - 分布式存储： Hadoop
  - 分布式计算： Hadoop、Spark
  - 视图存储数据库
    - nosql(HBase/Cassandra)
    - Redis/memcache
    - MySQL
- 实时处理层
  - 流式处理, 持续计算
  - 存储和分析某个窗口期内的数据（一段时间的热销排行，实时热搜等）
  - 实时数据收集 flume & kafka
  - 实时数据分析 spark streaming/storm/flink
- 服务层
  - 支持随机读
  - 需要在非常短的时间内返回结果
  - 读取批处理层和实时处理层结果并对其归并





#### Lambda架构图






### 推荐算法架构

#### 1,召回阶段 (海选)  recall

召回决定了最终推荐结果的天花板

##### 常用算法

- 协同过滤 cf  (Collaborative Filtering)
- 基于内容 cb (content based)



#### 2.排序阶段 （精选）ranking

- 召回决定了最终推荐结果的天花板, 排序逼近这个极限, 决定了最终的推荐效果

  CTR预估 (点击率预估 使用LR算法) 估计用户是否会点击某个商品 需要用户的点击数据

  逻辑回归 LogisticRegression ,无法根据特征

  0,1,估计用户点击某一个item概率

  根据概率去排序



#### 3.策略调整

- 过滤掉用户看了多次没反应的
- 商业合作,广告置顶,竞价排名
- 加入特征:时间点,价格区间




##### SPARK（计算引擎）

Apache Spark 是专为大规模数据处理而设计的快速通用的计算引擎。Spark是UC Berkeley AMP lab (加州大学伯克利分校的AMP实验室)所开源的类Hadoop MapReduce的通用并行框架，Spark，拥有Hadoop MapReduce所具有的优点；但不同于MapReduce的是——Job中间输出结果可以保存在内存中，从而不再需要读写HDFS，因此Spark能更好地适用于数据挖掘与机器学习等需要迭代的MapReduce的算法。



Spark 是一种与 Hadoop 相似的开源集群计算环境，但是两者之间还存在一些不同之处，这些有用的不同之处使 Spark 在某些工作负载方面表现得更加优越，换句话说，Spark 启用了内存分布数据集，除了能够提供交互式查询外，它还可以优化迭代工作负载。





#### 推荐模型构建流程

类似机器学习流程

数据处理-->特征工程-->训练算法模型-->评估上线

Data(数据)->Features(特征)->ML Algorithm(选择算法训练模型)->Prediction Output(预测输出)

- 数据清洗/数据处理
  - 数据来源
    - 显性数据
      - Rating 打分
      - Comments 评论/评价
    - 隐形数据
      -  Order history 历史订单
      -  Cart events 加购物车
      -  Page views 页面浏览
      -  Click-thru 点击
      -  Search log 搜索记录
  - 数据量/数据能否满足要求
- 特征工程
  - 从数据中筛选特征
    - 一个给定的商品，可能被拥有类似品味或需求的用户购买
    - 使用用户行为数据描述商品



#### 推荐系统算法



##### 1.基于协同过滤的推荐 CF



##### 2.基于内容的推荐 CB



##### 3.基于人口统计学的推荐



##### 4.混合推荐







### 协同过滤算法 CF

（Collaborative Filtering）

#### 基于近邻的协同过滤Memory-Based 

Memory-Based CF



Memory-Based利用用户行为数据计算相似度,要求数据量大,而Model_Based可以是稀疏矩阵

基于用户的协同过滤

基于物品的协同过滤

特征工程 需要把准备好

计算相似度

 - 杰卡德 01
 - 余弦/皮尔逊 连续的评分数据

根据相似度找到最相似的用户/最相似的物品

协同过滤对用户行为数据量要求比较高,数量量大时效果好



##### 算法思想：

基本的协同过滤推荐算法基于以下假设：

- “跟你喜好相似的人喜欢的东西你也很有可能喜欢” ：基于用户的协同过滤推荐（User-based CF）人以群分
- “跟你喜欢的东西相似的东西你也很有可能喜欢 ”：基于物品的协同过滤推荐（Item-based CF）物以类聚

实现协同过滤推荐有以下几个步骤：

1. 找出最相似的人或物品：TOP-N相似的人或物品

通过计算两两的相似度来进行排序，即可找出TOP-N相似的人或物品

1. 根据相似的人或物品产生推荐结果

利用TOP-N结果生成初始推荐结果，然后过滤掉用户已经有过记录的物品或明确表示不感兴趣的物品

以下是一个简单的示例，数据集相当于一个用户对物品的购买记录表：打勾表示用户对物品的有购买记录

- 关于相似度计算这里先用一个简单的思想：如有两个同学X和Y，X同学爱好[足球、篮球、乒乓球]，Y同学爱好[网球、足球、篮球、羽毛球]，可见他们的共同爱好有2个，那么他们的相似度可以用：2/3 * 2/4 = 1/3 ≈ 0.33 来表示。



 欧氏距离的值是一个非负数, 最大值正无穷, 通常计算相似度的结果希望是[-1,1]或[0,1]之间,一般可以使用




##### 相似度与相似距离

各个相似度对比

余弦相似度与方向相关于距离无关,应对方案:皮尔逊

余弦相似度与皮尔逊相关系数Pearson不适合计算布尔值(01分布,买与不买)向量之间的相关度

01分布用杰卡德相似度



##### 杰卡德相似性度量

（1）杰卡德相似系数

两个集合A和B交集元素的个数在A、B并集中所占的比例，称为这两个集合的杰卡德系数，用符号 J(A,B) 表示。杰卡德相似系数是衡量两个集合相似度的一种指标（余弦距离也可以用来衡量两个集合的相似度）。



（2）杰卡德距离(不相似的)

与杰卡德相似系数相反的概念是杰卡德距离（Jaccard Distance），可以用如下公式来表示：



**杰卡德距离 +  杰卡德相似系数 = 1**







pandas中corr方法可直接用于计算皮尔逊相关系数

计算相似度可以进行小**优化**,(对角)减少计算量,eg.九九乘法表的梯度



##### 皮尔逊相关系数Pearson

- 实际上也是余弦相似度, 不过先对向量做了中心化, 向量a b各自减去向量的均值后, 再计算余弦相似度
- 皮尔逊相似度计算结果在-1,1之间 -1表示负相关, 1表示正相关
- 度量两个变量是不是同增同减
- 皮尔逊相关系数度量的是两个变量的变化趋势是否一致, **不适合计算布尔值向量之间的相关度**





##### 余弦距离和欧氏距离的对比

余弦距离使用两个向量夹角的余弦值作为衡量两个个体间差异的大小。相比欧氏距离，余弦距离更加注重两个向量在方向上的差异。

借助三维坐标系来看下欧氏距离和余弦距离的区别：



从上图可以看出，欧氏距离衡量的是空间各点的绝对距离，跟各个点所在的位置坐标直接相关；而余弦距离衡量的是空间向量的夹角，更加体现在方向上的差异，而不是位置。如果保持A点位置不变，B点朝原方向远离坐标轴原点，那么这个时候余弦距离 



 是保持不变的（因为夹角没有发生变化），而A、B两点的距离显然在发生改变，这就是欧氏距离和余弦距离之间的不同之处。

欧氏距离和余弦距离各自有不同的计算方式和衡量特征，因此它们适用于不同的数据分析模型：

欧氏距离能够体现个体数值特征的绝对差异，所以更多的用于需要从维度的数值大小中体现差异的分析，如使用用户行为指标分析用户价值的相似度或差异。

余弦距离更多的是从方向上区分差异，而对绝对的数值不敏感，更多的用于使用用户对内容评分来区分兴趣的相似度和差异，同时修正了用户间可能存在的度量标准不统一的问题（因为余弦距离对绝对数值不敏感）。

 



预测用户对物品的评分 （以用户1对电影1评分为例）
$$
评分公式  pred(u,i)=r^ui=∑v∈Usim(u,v)∗rvi∑v∈U|sim(u,v)|
$$









##### User_CF和Item_CF比较

User_CF适合信息流方向,内容更新极快,例如新闻,小视频,直播



##### 协同过滤推荐优缺点






#### 推荐评估

准确性

- rmse,mae精准率,召回率
- 业务指标是否提升

##### 指标

- 准确度

- 召回率
- 覆盖率
- 多样性



##### 覆盖率与熵



##### 在线评估

灰度发布(低比例测试,慢慢调高比例)

A/B测试(对照组)



随机推荐,热门推荐

指标与业绩平衡





#### 推荐系统的冷启动问题

##### 用户冷启动(新用户)



方案:

1. 尽可能收集各种用户的信息,给用户打各种标签
2. 通过标签做用户的聚类,通过user_cf推荐
3. 可以考虑使用热门推荐



手机电量状况与心理状态

热门推荐与随机推荐



##### 物品冷启动



利用标签找到相似的物品,相似的物品可能有消费记录

基于内容的推荐





eg.推荐电影,推荐汽车,推荐房产,推荐股票,推荐贷款产品



##### 系统冷启动(新用户 +　新物品）



- 基于内容的推荐 系统早期
- 基于内容的推荐逐渐过渡到协同过滤
- 基于内容的推荐和协同过滤的推荐结果都计算出来 加权求和得到最终推荐结果



### 基于模型的协同过滤Model-Based

Model-Based CF算法

Model_Based可以是稀疏矩阵,Memory-Based利用用户行为数据计算相似度,要求数据量大,



- 基于分类算法、回归算法、聚类算法
- 基于矩阵分解的推荐
- 基于神经网络算法
- 基于图模型算法



- 奇异值分解（SVD）
- 潜在语义分析（LSA）
- 支撑向量机（SVM)



#### 1-基于回归模型的协同过滤推荐

Baseline

梯度下降推导

使用Baseline的算法思想预测评分的步骤如下：

- 计算所有电影的平均评分μμ（即全局平均评分）
- 计算每个用户评分与平均评分μ的偏置值buμ的偏置值bu
- 计算每部电影所接受的评分与平均评分μ的偏置值biμ的偏置值bi
- 预测用户对电影的评分： 
$$
r_u,_i=bui=μ+bu+bi 
$$

偏导数:

在数学中，一个多变量的函数的偏导数，就是它关于其中一个变量的导数而保持其他变量恒定（相对于全导数，在其中所有变量都允许变化）。偏导数在[向量分析](https://baike.baidu.com/item/%E5%90%91%E9%87%8F%E5%88%86%E6%9E%90/10564843)和[微分](https://baike.baidu.com/item/%E5%BE%AE%E5%88%86)几何中是很有用的。



data_xx.itertuples()



相关代码:

```python
ratings.groupby('userId') #返回groupby对象
ratings.groupby('userId').any() #聚合 

ratings.groupby('userId').agg()
    
np.random.shuffle(index)    # 打乱列表
```



```python
语法 zip([iterable, ...])

示例：

a = [1,2,3]
b = [4,5,6]
c = [4,5,6,7,8]
zipped = zip(a,b)     # 返回一个对象
>>> zipped
<zip object at 0x103abc288>
>>> list(zipped)  # list() 转换为列表
[(1, 4), (2, 5), (3, 6)]
>>> list(zip(a,c))              # 元素个数与最短的列表一致
[(1, 4), (2, 5), (3, 6)]

 # 与 zip 相反，zip(*) 可理解为解压，返回二维矩阵式
a1, a2 = zip(*zip(a,b))         
>>> list(a1)
[1, 2, 3]
>>> list(a2)
[4, 5, 6]
```



核心代码:

```python
#更新bu,bi
#number_epochs 迭代次数,alpha学习率,reg正则化系统
for i in range(number_epochs):
    print('inter%d'%i)
    for uid,iid,real_rating in dataset.itertuples(index=False):
        #差值(样本损失)  error = 真实值 - 预测值
        error =real_rating -(global_mean +bu[uid] +bi[iid])
        # 梯度下降法推导
        # bu  = bu+α∗(∑u,i∈R(rui−μ−bu−bi)−λ∗bu) 
        # 随机梯度下降
        # bu = bu + a*(error - λ∗bu)
        bu[uid] += alpha *(error -reg*bu[uid])
        bi[iid] += alpha *(error -reg*bi[iid])
```





##### 1-1随机梯度下降优化

```python
import pandas as pd
import numpy as np

def data_split(data_path, x=0.8, random=False):
    '''
    切分数据集， 这里为了保证用户数量保持不变，将每个用户的评分数据按比例进行拆分
    :param data_path: 数据集路径
    :param x: 训练集的比例，如x=0.8，则0.2是测试集
    :param random: 是否随机切分，默认False
    :return: 用户-物品评分矩阵
    '''
    print("开始切分数据集...")
    # 设置要加载的数据字段的类型
    dtype = {"userId": np.int32, "movieId": np.int32, "rating": np.float32}
    # 加载数据，我们只用前三列数据，分别是用户ID，电影ID，已经用户对电影的对应评分
    ratings = pd.read_csv(data_path, dtype=dtype, usecols=range(3))

    testset_index = []
    # 为了保证每个用户在测试集和训练集都有数据，因此按userId聚合
    for uid in ratings.groupby("userId").any().index:
        user_rating_data = ratings.where(ratings["userId"]==uid).dropna()
        if random:
            # 因为不可变类型不能被 shuffle方法作用，所以需要强行转换为列表
            index = list(user_rating_data.index)
            np.random.shuffle(index)    # 打乱列表
            _index = round(len(user_rating_data) * x)
            testset_index += list(index[_index:])
        else:
            # 将每个用户的x比例的数据作为训练集，剩余的作为测试集
            index = round(len(user_rating_data) * x)
            testset_index += list(user_rating_data.index.values[index:])

    testset = ratings.loc[testset_index]
    trainset = ratings.drop(testset_index)
    print("完成数据集切分...")
    return trainset, testset

def accuray(predict_results, method="all"):
    '''
    准确性指标计算方法
    :param predict_results: 预测结果，类型为容器，每个元素是一个包含uid,iid,real_rating,pred_rating的序列
    :param method: 指标方法，类型为字符串，rmse或mae，否则返回两者rmse和mae
    :return:
    '''

    def rmse(predict_results):
        '''
        rmse评估指标
        :param predict_results:
        :return: rmse
        '''
        length = 0
        _rmse_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
        return round(np.sqrt(_rmse_sum / length), 4)

    def mae(predict_results):
        '''
        mae评估指标
        :param predict_results:
        :return: mae
        '''
        length = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _mae_sum += abs(pred_rating - real_rating)
        return round(_mae_sum / length, 4)

    def rmse_mae(predict_results):
        '''
        rmse和mae评估指标
        :param predict_results:
        :return: rmse, mae
        '''
        length = 0
        _rmse_sum = 0
        _mae_sum = 0
        for uid, iid, real_rating, pred_rating in predict_results:
            length += 1
            _rmse_sum += (pred_rating - real_rating) ** 2
            _mae_sum += abs(pred_rating - real_rating)
        return round(np.sqrt(_rmse_sum / length), 4), round(_mae_sum / length, 4)

    if method.lower() == "rmse":
        rmse(predict_results)
    elif method.lower() == "mae":
        mae(predict_results)
    else:
        return rmse_mae(predict_results)
#随机梯度下降 封装
class BaselineCFBySGD(object):
    def __init__(self,number_epochs,alpha,reg,columns=['uid','iid','rating']):
        
        # 梯度下降最高迭代次数
        self.number_epochs = number_epochs
        # 学习率
        self.alpha = alpha
        # 正则参数
        self.reg = reg
        # 数据集中user-item-rating字段的名称
        self.columns = columns
        
    
    def fit(self,dataset):
        self.dataset = dataset
        #用户评分数据
        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        #用户评分数据
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]
        # 计算全局平均分
        self.global_mean = self.dataset[self.columns[2]].mean()
        # 调用sgd方法训练模型参数
        self.bu, self.bi = self.sgd()
        
    def sgd(self):
                '''
        利用随机梯度下降，优化bu，bi的值
        :return: bu, bi
        
        '''
        # 初始化bu、bi的值，全部设为0
        bu = dict(zip(self.users_ratings.index, np.zeros(len(self.users_ratings))))
        bi = dict(zip(self.items_ratings.index, np.zeros(len(self.items_ratings))))
        
        #更新bu,bi
        #number_epochs 迭代次数,alpha学习率,reg正则化系统
        for i in range(number_epochs):
            print('inter%d'%i)
            for uid,iid,real_rating in dataset.itertuples(index=False):
                #差值(样本损失)  error = 真实值 - 预测值
                error =real_rating -(global_mean +bu[uid] +bi[iid])
                # 梯度下降法推导
                # bu  = bu+α∗(∑u,i∈R(rui−μ−bu−bi)−λ∗bu) 
                # 随机梯度下降
                # bu = bu + a*(error - λ∗bu)
                bu[uid] += alpha *(error -reg*bu[uid])
                bi[iid] += alpha *(error -reg*bi[iid])
        return bu,bi
    
    #预测
    def predict(self,uid,iid):
        predict_rating =self.global_mean + self.bu[uid] + self.bi[iid]
        
        return predict_rating
    

#调用
if __name__ == '__main__':
    trainset, testset = data_split("../../data/ml-latest-small/ratings.csv", random=True)

    bcf = BaselineCFBySGD(20, 0.1, 0.1, ["userId", "movieId", "rating"])
    bcf.fit(trainset)

    pred_results = bcf.test(testset)

    rmse, mae = accuray(pred_results)

    print("rmse: ", rmse, "mae: ", mae)
```







针对回归问题的评估:均方根误差rmse,平均绝对误差mae



##### 1-2 ALS交替最小二乘法优化

算某个系数时,默认另外为已知



- 迭代更新bu bi

```python
for i in range(number_epochs):
    print("iter%d" % i)
    for iid, uids, ratings in items_ratings.itertuples(index=True):
        _sum = 0
        for uid, rating in zip(uids, ratings):
            _sum += rating - global_mean - bu[uid]
        bi[iid] = _sum / (reg_bi + len(uids))

    for uid, iids, ratings in users_ratings.itertuples(index=True):
        _sum = 0
        for iid, rating in zip(iids, ratings):
            _sum += rating - global_mean - bi[iid]
        bu[uid] = _sum / (reg_bu + len(iids))
```





#### 2-基于矩阵分解的协同过滤推荐 !!

##### surprise库

Surprise · A Python scikit for recommender systems.

<http://surpriselib.com/>

<https://github.com/NicolasHug/Surprise>






##### Traditional SVD



 SVD 适用于稠密矩阵,评分系统一般为稀疏矩阵,即不适合用SVD



##### FunkSVD (LFM)

适用于**稀疏矩阵**

用户隐语义模型矩阵

- 用户向量

物品隐主义模型矩阵

- 物品向量

在**spark**中有该功能的封装



刚才提到的Traditional SVD首先需要填充矩阵，然后再进行分解降维，同时存在计算复杂度高的问题，因为要分解成3个矩阵，所以后来提出了Funk SVD的方法，它不在将矩阵分解为3个矩阵，而是分解为2个用户-**隐含特征**，项目-隐含特征的矩阵，Funk SVD也被称为最原始的LFM模型



借鉴线性回归的思想，通过最小化观察数据的平方来寻求最优的用户和项目的隐含向量表示。同时为了避免过度拟合（Overfitting）观测数据，又提出了带有L2正则项的FunkSVD，上公式：



以上两种最优化函数都可以通过梯度下降或者随机梯度下降法来寻求最优解。





##### LFM梯度下降代码实现

```python
# 评分矩阵R
R = np.array([[4,0,2,0,1],
             [0,2,3,0,0],
             [1,0,2,4,0],
             [5,0,0,3,1],
             [0,0,1,5,1],
             [0,3,2,4,1],])

"""
@输入参数：
R：M*N 的评分矩阵
K：隐特征向量维度
max_iter: 最大迭代次数
alpha：步长
lamda：正则化系数
@输出：
分解之后的 P，Q
P：初始化用户特征矩阵M*K
Q：初始化物品特征矩阵N*K
"""

# 给定超参数
K = 5
max_iter = 5000
alpha = 0.0002
lamda = 0.004

# 核心算法
def LFM_grad_desc( R, K=2, max_iter=1000, alpha=0.0001, lamda=0.002 ):
    # 基本维度参数定义
    M = len(R)
    N = len(R[0])
    
    # P,Q初始值，随机生成
    P = np.random.rand(M, K)
    Q = np.random.rand(N, K)
    Q = Q.T #转置
    
    # 开始迭代
    for step in range(max_iter):
        # 对所有的用户u、物品i做遍历，对应的特征向量Pu、Qi梯度下降
        for u in range(M):
            for i in range(N):
                # 对于每一个大于0的评分，求出预测评分误差
                if R[u][i] > 0:
                    eui = np.dot( P[u,:], Q[:,i] ) - R[u][i]
                    
                    # 代入公式，按照梯度下降算法更新当前的Pu、Qi
                    for k in range(K):
                        P[u][k] = P[u][k] - alpha * ( 2 * eui * Q[k][i] + 2 * lamda * P[u][k] )
                        Q[k][i] = Q[k][i] - alpha * ( 2 * eui * P[u][k] + 2 * lamda * Q[k][i] )
        
        # u、i遍历完成，所有特征向量更新完成，可以得到P、Q，可以计算预测评分矩阵
        predR = np.dot( P, Q )
        
        # 计算当前损失函数
        cost = 0
        for u in range(M):
            for i in range(N):
                if R[u][i] > 0:
                    cost += ( np.dot( P[u,:], Q[:,i] ) - R[u][i] ) ** 2
                    # 加上正则化项
                    for k in range(K):
                        cost += lamda * ( P[u][k] ** 2 + Q[k][i] ** 2 )
        if cost < 0.0001:
            break
        
    return P, Q.T, cost


#测试
P, Q, cost = LFM_grad_desc(R, K, max_iter, alpha, lamda)

print(P)
print(Q)
print(cost)

predR = P.dot(Q.T)

print(R)
print(predR)
```



黑马代码LFM

```python
#数据加载
import pandas as pd
import numpy as np


dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
dataset = pd.read_csv("ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))

#数据初始化

# 用户评分数据  groupby 分组  groupby('userId') 根据用户id分组 agg（aggregation聚合）
users_ratings = dataset.groupby('userId').agg([list])
# 物品评分数据
items_ratings = dataset.groupby('movieId').agg([list])
# 计算全局平均分
global_mean = dataset['rating'].mean()
# 初始化P Q  610  9700   K值  610*K    9700*K
# User-LF  10 代表 隐含因子个数是10个
P = dict(zip(users_ratings.index,np.random.rand(len(users_ratings),10).astype(np.float32)
        ))
# Item-LF
Q = dict(zip(items_ratings.index,np.random.rand(len(items_ratings),10).astype(np.float32)
        ))

#梯度下降优化损失函数
for i in range(15):
    print('*'*10,i)
    for uid,iid,real_rating in dataset.itertuples(index = False):
        #遍历 用户 物品的评分数据 通过用户的id 到用户矩阵中获取用户向量
        v_puk = P[uid]
        # 通过物品的uid 到物品矩阵里获取物品向量
        v_qik = Q[iid]
        #计算损失
        error = real_rating-np.dot(v_puk,v_qik)
        # 0.02学习率 0.01正则化系数
        v_puk += 0.02*(error*v_qik-0.01*v_puk)
        v_qik += 0.02*(error*v_puk-0.01*v_qik)

        P[uid] = v_puk
        Q[iid] = v_qik
#评分预测
def predict(self, uid, iid):
    # 如果uid或iid不在，我们使用全剧平均分作为预测结果返回
    if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
        return self.globalMean
    p_u = self.P[uid]
    q_i = self.Q[iid]

    return np.dot(p_u, q_i)
'''
LFM Model
'''

# 评分预测    1-5
class LFM(object):

    def __init__(self, alpha, reg_p, reg_q, number_LatentFactors=10, number_epochs=10, columns=["uid", "iid", "rating"]):
        self.alpha = alpha # 学习率
        self.reg_p = reg_p    # P矩阵正则
        self.reg_q = reg_q    # Q矩阵正则
        self.number_LatentFactors = number_LatentFactors  # 隐式类别数量
        self.number_epochs = number_epochs    # 最大迭代次数
        self.columns = columns

    def fit(self, dataset):
        '''
        fit dataset
        :param dataset: uid, iid, rating
        :return:
        '''

        self.dataset = pd.DataFrame(dataset)

        self.users_ratings = dataset.groupby(self.columns[0]).agg([list])[[self.columns[1], self.columns[2]]]
        self.items_ratings = dataset.groupby(self.columns[1]).agg([list])[[self.columns[0], self.columns[2]]]

        self.globalMean = self.dataset[self.columns[2]].mean()

        self.P, self.Q = self.sgd()

    def _init_matrix(self):
        '''
        初始化P和Q矩阵，同时为设置0，1之间的随机值作为初始值
        :return:
        '''
        # User-LF
        P = dict(zip(
            self.users_ratings.index,
            np.random.rand(len(self.users_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        # Item-LF
        Q = dict(zip(
            self.items_ratings.index,
            np.random.rand(len(self.items_ratings), self.number_LatentFactors).astype(np.float32)
        ))
        return P, Q

    def sgd(self):
        '''
        使用随机梯度下降，优化结果
        :return:
        '''
        P, Q = self._init_matrix()

        for i in range(self.number_epochs):
            print("iter%d"%i)
            error_list = []
            for uid, iid, r_ui in self.dataset.itertuples(index=False):
                # User-LF P
                ## Item-LF Q
                v_pu = P[uid] #用户向量
                v_qi = Q[iid] #物品向量
                err = np.float32(r_ui - np.dot(v_pu, v_qi))

                v_pu += self.alpha * (err * v_qi - self.reg_p * v_pu)
                v_qi += self.alpha * (err * v_pu - self.reg_q * v_qi)

                P[uid] = v_pu 
                Q[iid] = v_qi

                # for k in range(self.number_of_LatentFactors):
                #     v_pu[k] += self.alpha*(err*v_qi[k] - self.reg_p*v_pu[k])
                #     v_qi[k] += self.alpha*(err*v_pu[k] - self.reg_q*v_qi[k])

                error_list.append(err ** 2)
            print(np.sqrt(np.mean(error_list)))
        return P, Q

    def predict(self, uid, iid):
        # 如果uid或iid不在，我们使用全剧平均分作为预测结果返回
        if uid not in self.users_ratings.index or iid not in self.items_ratings.index:
            return self.globalMean

        p_u = self.P[uid]
        q_i = self.Q[iid]

        return np.dot(p_u, q_i)

    def test(self,testset):
        '''预测测试集数据'''
        for uid, iid, real_rating in testset.itertuples(index=False):
            try:
                pred_rating = self.predict(uid, iid)
            except Exception as e:
                print(e)
            else:
                yield uid, iid, real_rating, pred_rating

if __name__ == '__main__':
    dtype = [("userId", np.int32), ("movieId", np.int32), ("rating", np.float32)]
    dataset = pd.read_csv("datasets/ml-latest-small/ratings.csv", usecols=range(3), dtype=dict(dtype))

    lfm = LFM(0.02, 0.01, 0.01, 10, 100, ["userId", "movieId", "rating"])
    lfm.fit(dataset)

    while True:
        uid = input("uid: ")
        iid = input("iid: ")
        print(lfm.predict(int(uid), int(iid)))
```





##### LFM总结

LFM(FunkSVD)实现流程(重点)

- LFM(FunkSVD)梯度下降推导
  - 利用平方差来构建损失函数, 加入正则化项
  - 梯度下降更新参数 puk和qik
    - 损失函数: 分别对puk和qik求偏导,
    - 随机梯度, 去掉最外部的求和, 得出puk和qik梯度下降公式
- 实现思路:
  - 加载数据
  - 数据初始
  - 梯度下降优化损失函数(更新puk, qik的值)
  - 评分预测

- LFM(FunkSVD)基本原理
  - 将用户-物品评分矩阵拆分为两个小矩阵, 用户隐因子矩阵, 物品隐因子矩阵
  - 用户隐因子矩阵(P) 每一个维度代表了会影响到用户对物品评分的用户特征
  - 物品隐因子矩阵(Q) 每一个维度代表了会影响到物品得分的特征
  - 从用户隐因子矩阵中取出用户向量, 从物品隐因子矩阵矩阵中取出物品向量, 点积即可得到用户对该物品的评分预测
  - 利用优化损失函数的思想求解两个矩阵, 可以采用梯度下降的方法
- LFM(FunkSVM)的梯度下降优化
  - 初始化两个矩阵P和Q, 隐因子个数需要人为设定
  - 向量相乘使用numpy的dot函数直接计算



##### BiasSVD:

在LFM加入Baseline思想 ,引入用户偏置,物品偏置

在FunkSVD提出来之后，出现了很多变形版本，其中一个相对成功的方法是BiasSVD，顾名思义，即带有偏置项的SVD分解：



它基于的假设和Baseline基准预测是一样的，但这里将Baseline的偏置引入到了矩阵分解中







##### SVD++:

人们后来又提出了改进的BiasSVD，被称为SVD++，该算法是在BiasSVD的基础上添加了用户的隐式反馈信息：



显示反馈指的用户的评分这样的行为，隐式反馈指用户的浏览记录、购买记录、收听记录等。

SVD++是基于这样的假设：在BiasSVD基础上，认为用户对于项目的历史浏览记录、购买记录、收听记录等可以从侧面反映用户的偏好。



##### 协同过滤VS隐语义



##### 隐语义

即基于模型的协同过滤 ModelBasedCF

隐含因子



##### 评估标准

- 准确度

- 召回率
- 覆盖率
- 多样性




### 基于内容的推荐算法 CB

（Content-Based）

#### 简介

在发现用户新兴趣方面不如协同过滤(CF)



物品打标签,物品画像

有用户行为数据

- 用户行为画像
- 建立标签对物品的倒排索引
- 用户画像标签找对应的物品
  - 根据用户对标签的消费次数
  - 用户对标签消费时的评分

可以解决冷启动





基于内容的推荐方法是非常直接的，它以物品的内容描述信息为依据来做出的推荐，本质上是基于对物品和用户自身的特征或属性的直接分析和计算。

例如，假设已知电影A是一部喜剧，而恰巧我们得知某个用户喜欢看喜剧电影，那么我们基于这样的已知信息，就可以将电影A推荐给该用户。



##### 基于内容的推荐

- PGC生成标签

- 从描述性的文字中提取关键词
  - Tf_idf    , textrank
- word2vec 词向量
  - 词-->向量
- doc2vec
  - 文档-->向量





##### 算法流程：

- 建立物品画像
- 有用户行为数据
  - 建立用户画像
    - 用户消费过哪些物品 这些物品的标签就可以打到用户身上
  - 建立倒排索引
    - 物品找标签 倒排就是根据标签找物品
  - 根据用户的画像中的标签 找到标签对应的所有物品
    - 排序可以根据用户对标签的消费次数
    - 也可以根据用户对标签的打分情况
- 没有用户行为数据情况 系统冷启动
  - 建立物品画像 系统生成
  - 从描述性的文字中提取关键词
    - tf-idf textrank
  - word2vec/doc2vec
  - 把物品的描述-》转换成一组关键词-》转换成一个文档向量
  - 文档向量的相似度 就可以表示内容的相似性
- 可以为每一件商品创建文档向量 计算和其它物品的相似度 召回相似度最高的前n个商品



- 根据PGC/UGC内容构建物品画像
  - 建立倒排索引,可以根据标签找物品
- 根据用户行为记录生成用户画像
- 根据用户画像从物品中寻找最匹配的TOP-N物品进行推荐



##### 物品冷启动处理：

没用用户行为数据情况,系统冷启动

- 根据PGC内容构建物品画像
- 利用物品画像计算物品间两两相似情况
- 为每个物品产生TOP-N最相似的物品进行相关推荐：如与该商品相似的商品有哪些？与该文章相似文章有哪些？



#### 用户画像

- 可以用于风控,减少羊毛党的推荐,减少对差评党的推荐
- 标签可视化
- 用户画像可视化
- eg.年度账单总结
- eg.芝麻分维度分析




##### 标签体系



##### 验证

- 准确性验证


##### 用户画像生产和应用:逻辑架构



##### 企业用户触点



#### 内容推荐算法的原理:

1. 将产品分解为一系列标签。例如,一个手机产品的标签可以包括品牌、价格、产地、颜色、款式等。如果是自营b2c电商,一般可以在产品入库时手动打标签。

   2.基于用户行为(浏览、购买、收藏)计算每个用户的产品兴趣标签。例如,用户购买了一个产品,则将该产品的所有标签赋值给该用户,每个标签打分为1;用户浏览了一个产品,则将该产品的所有标签赋值给该用户,每个标签打分为0.5。计算复杂度为:已有产品数量*用户量。该过程为离线计算。

2. 针对所有新产品,分别计算每个用户的产品标签与每个新产品的相似度(基于cosine similarity)。计算复杂度为:新产品数量*用户量。该过程为在线计算。



从可行性角度,一个应用场景**是否适合**用内容推荐算法取决于:

1. 是否可以持续为产品打标签。

2. 标签是否可以覆盖产品的核心属性?例如,手机产品的标签一般可以覆盖消费者购物的核心决策因素,但是女装一般比较难(视觉效果很难被打标)。

##### **内容推荐算法的优势:**

1. 推荐结果可理解:不仅每个用户的核心兴趣点可以被标签化(便于理解每个用户的兴趣),并且可以在每一个推荐结果的展示中现实标签,便于消费者理解推荐结果(如下图红框)。


2. 推荐结果稳定性强:对于用户行为不丰富的产品类型(例如,金融产品),协同过滤很难找到同兴趣用户群或关联产品,在相似度计算中稀疏度太高。然而,内容推荐主要使用标签,标签对用户兴趣捕捉稳定性要远远高于单个产品。

3. 便于人机协作:用户可以勾选或者关注推荐标签,从而通过自己的操作来发现自己的个性化需求。



##### 内容推荐算法的劣势:

1. 不适合发现惊喜:如果一个产品不易于被标签穷举或描述产品的标签还没出现,则该产品很难被准确推荐。

2. 在线应用计算复杂度较高:需要基于每个用户来计算相似产品。



##### 基于内容推荐(CB) 与 Item_CF 的区别

向量来源不同:

Item_CF 需要用户对物品评分的数据

CB 通过文本检索技术把词/文档转成向量,可以不需要用户参与,即不受限用户行为数据

CB基于文本分析

- 基于内容推荐解决系统冷启动问题套路
  - 文本检索技术 把词/文档转换成了向量
  - 物品-》向量描述-》计算向量的相似度-》把和当前物品相似度高的内容推荐出去
  - 不受限与用户行为数据
- 协同过滤 基于物品的协同过滤
  - 用户-物品 评分矩阵 -> 向量描述
  - 物品-》向量描述-》计算向量的相似度-》把和当前物品相似度高的内容推荐出去

### Hadoop

- 重点

- HDFS的使用 ☆☆☆☆☆
  - 启动
  - 文件的上传下载删除
- MapReduce的原理 ☆☆☆☆
  - Map
  - Reduce
- MRJob☆☆☆
  - python开发MapReduce
  - 继承MRJob  mapper  reducer
  - 练习WordCount案例
- HDFS的架构以及读写流程
  - NameNode
  - DataNode



#### Hadoop的概念

- Apache™ Hadoop® 是一个开源的,可靠的(reliable),可扩展的(scalable)分布式计算框架
  - 分布式计算框架:允许使用简单的编程模型跨计算机集群分布式处理大型数据集
  - **可扩展**: 从单个服务器扩展到数万台计算机，每台计算机都提供本地计算和存储
  - **可靠的**: 不依靠硬件来提供高可用性(high-availability)，而是在应用层检测和处理故障，从而在计算机集群之上提供高可用服务

##### Hadoop用途

- 搭建大型数据仓库
  - 保存数据的历史版本
- PB级数据的存储 处理 分析 统计等业务
  - 搜索引擎
  - 日志分析(日活,消量,新增用户,7日留存,月留存)
  - 数据挖掘(挖掘=分析 + 预测)
  - 商业智能(Business Intelligence，简称：BI)

##### 发展史

- 2003-2004年 Google发表了三篇论文
  - GFS：Google的分布式文件系统Google File System
  - [MapReduce](https://en.wikipedia.org/wiki/MapReduce): Simplified Data Processing on Large Clusters
  - BigTable：一个大型的分布式数据库
- 2006年2月Hadoop成为Apache的独立开源项目( Doug Cutting等人实现了DFS和MapReduce机制)。

- 
- 搜索引擎时代
  - 有保存大量网页的需求(单机 集群)
  - 词频统计 word count PageRank
- 数据仓库时代
  - FaceBook推出**Hive**
  - 曾经进行数分析与统计时, 仅限于数据库,受数据量和计算能力的限制, 我们只能对最重要的数据进行统计和分析(决策数据,财务相关)
  - Hive可以在Hadoop上运行SQL操作, 可以把运行日志, 应用采集数据,数据库数据放到一起分析
- 数据挖掘时代
  - 啤酒尿不湿
  - 关联分析
  - 用户画像/物品画像
- 机器学习时代 广义大数据
  - 大数据提高数据存储能力, 为机器学习提供燃料
  - alpha go
  - siri 小爱同学 天猫精灵

#####  Hadoop优势

- 高可靠
  - 数据存储: 数据块多副本
  - 数据计算: 某个节点崩溃, 会自动重新调度作业计算
- 高扩展性
  - 存储/计算资源不够时，可以横向的线性扩展机器
  - 一个集群中可以包含数以千计的节点
  - 集群可以使用廉价机器，成本低
- Hadoop生态系统成熟
  - 有开源的功能模块,改装一下就能应用





##### Hadoop组件

- MapReduce
- YARN
- HDFS



#### MapReduce分布式计算框架

A YARN-based system for parallel processing of large data sets.

- 分布式**计算框架**,Map拆分数据并计算,Reduce汇整每一部分的计算结果
- 源于Google的MapReduce论文，论文发表于2004年12月
- MapReduce是GoogleMapReduce的开源实现
- MapReduce特点:扩展性&容错性&海量数据离线处理
- 移动计算比移动数据要划算



##### mrjob 简介

- 使用python开发在Hadoop上运行的程序, mrjob是最简单的方式
- mrjob程序可以在本地测试运行也可以部署到Hadoop集群上运行
- 如果不想成为hadoop专家, 但是需要利用Hadoop写MapReduce代码,mrJob是很好的选择



##### MRJob实现MapReduce

- hadoop 提供了一个hadoop streaming的jar包， 通过hadoop streaming 可以用python 脚本写mapreduce任务 ， hadoop streaming 做用帮助把脚本翻译成java, 使用hadoop streaming有些麻烦
  - map阶段对应一个python文件
  - reduce阶段对应一个python文件
- MRJob用法
  - 创建一个类继承MRJob
  - 重写 mapper 和 reducer
  - 如果有多个map 和reduce 阶段 需要创建MRStep对象
  - 创建MRStep对象 可以指定每一个阶段的mapper对应的方法，reducer对应的方法，combiner对应的方法
  - 通过重写steps方法 返回MRStep的list 指定多个step的执行顺序



##### mrjob实现WordCount

```python
from mrjob.job import MRJob

class MRWordCount(MRJob):

    #每一行从line中输入
    def mapper(self, _, line):
        for word in line.split():
            yield word,1

    # word相同的 会走到同一个reduce
    def reducer(self, word, counts):
        yield word, sum(counts)

if __name__ == '__main__':
    MRWordCount.run()
```



##### mrjob实现topN统计

```python
import sys
from mrjob.job import MRJob,MRStep
import heapq

class TopNWords(MRJob):
    def mapper(self, _, line):
        if line.strip() != "":
            for word in line.strip().split():
                yield word,1

    #介于mapper和reducer之间，用于临时的将mapper输出的数据进行统计
    def combiner(self, word, counts):
        yield word,sum(counts)

    def reducer_sum(self, word, counts):
        yield None,(sum(counts),word)

    #利用heapq将数据进行排序，将最大的2个取出
    def top_n_reducer(self,_,word_cnts):
        for cnt,word in heapq.nlargest(2,word_cnts):
            yield word,cnt

    #实现steps方法用于指定自定义的mapper，comnbiner和reducer方法
    def steps(self):
        #传入两个step 定义了执行的顺序
        return [
            MRStep(mapper=self.mapper,
                   combiner=self.combiner,
                   reducer=self.reducer_sum),
            MRStep(reducer=self.top_n_reducer)
        ]

def main():
    TopNWords.run()

if __name__=='__main__':
    main()
```






##### MapReduce原理

**单机程序计算流程**

输入数据--->读取数据--->处理数据--->写入数据--->输出数据

**Hadoop计算流程**

input data：输入数据

InputFormat：对数据进行切分，格式化处理

map：将前面切分的数据做map处理(将数据进行分类，输出(k,v)键值对数据)

shuffle&sort:将相同的数据放在一起，并对数据进行排序处理

reduce：将map输出的数据进行hash计算，对每个map数据进行统计计算

OutputFormat：格式化输出数据






##### MapReduce架构

MapReduce架构 1.X

- JobTracker:负责接收客户作业提交，负责任务到作业节点上运行，检查作业的状态
- TaskTracker：由JobTracker指派任务，定期向JobTracker汇报状态，在每一个工作节点上永远只会有一个TaskTracker

MapReduce2.X架构

- ResourceManager：负责资源的管理，负责提交任务到NodeManager所在的节点运行，检查节点的状态
- NodeManager：由ResourceManager指派任务，定期向ResourceManager汇报状态



##### 计算速度慢的原因

- 进程模型,启动任务需启动JVM虚拟机,比较耗时

- 没有完全使用内存,会用磁盘读写



#### YARN资源调度系统

A framework for job scheduling and cluster resource management.(资源调度系统)

- YARN: Yet Another Resource Negotiator
  - **Hadoop 2.X以前没有YARN,使用Mesos开源框架实现**
- 负责整个集群资源的管理和调度
- YARN特点:扩展性&容错性&多框架资源统一调度



启动MapReduce需要先启动YARN



##### YARN架构

- ResourceManager: RM 资源管理器  整个集群同一时间提供服务的RM只有一个，负责集群资源的统一管理和调度  处理客户端的请求： submit, kill  监控我们的NM，一旦某个NM挂了，那么该NM上运行的任务需要告诉我们的AM来如何进行处理
  - 同一时间只能有一个
- NodeManager: NM 节点管理器  整个集群中有多个，负责自己本身节点资源管理和使用  定时向RM汇报本节点的资源使用情况  接收并处理来自RM的各种命令：启动Container  处理来自AM的命令
  - 同一时间可以有多个运行
- ApplicationMaster: AM  每个应用程序对应一个：MR、Spark，负责应用程序的管理  为应用程序向RM申请资源（core、memory），分配给内部task  需要与NM通信：启动/停止task，task是运行在container里面，AM也是运行在container里面
- Container 容器: 封装了CPU、Memory等资源的一个容器,是一个任务运行环境的抽象
- Client: 提交作业 查询作业的运行进度,杀死作业





##### 流程


1，Client提交作业请求

2，ResourceManager 进程和 NodeManager 进程通信，根据集群资源，为用户程序分配第一个Container(容器)，并将 ApplicationMaster 分发到这个容器上面

3，在启动的Container中创建ApplicationMaster

4，ApplicationMaster启动后向ResourceManager注册进程,申请资源

5，ApplicationMaster申请到资源后，向对应的NodeManager申请启动Container,将要执行的程序分发到NodeManager上

6，Container启动后，执行对应的任务

7，Tast执行完毕之后，向ApplicationMaster返回结果

8，ApplicationMaster向ResourceManager 请求kill





#### HDFS分布式文件系统

##### Hadoop Distributed File System (HDFS)

 A distributed file system that provides high-throughput access to application data.(分布式文件系统)

- 源自于Google的GFS论文, 论文发表于2003年10月
- HDFS是GFS的开源实现
- HDFS的特点:扩展性&容错性&海量数量存储
- 将文件切分成指定大小的数据块, 并在多台机器上保存多个副本
- 数据切分、多副本、容错等操作对用户是透明的



##### 命令

| 命令                     | 说明                                          |
| ------------------------ | --------------------------------------------- |
| hadoop fs -mkdir         | 创建HDFS目录                                  |
| hadoop fs -ls            | 列出HDFS目录                                  |
| hadoop fs -copyFromLocal | 使用-copyFromLocal复制本地文件（local）到HDFS |
| hadoop fs -put           | 使用-put复制本地（local）文件到HDFS           |
| hadoop fs -copyToLocal   | 将HDFS上的文件复制到本地（local）             |
| hadoop fs -get           | 将HDFS上的文件复制到本地（local）             |
| hadoop fs -cp            | 复制HDFS文件                                  |
| hadoop fs -rm            | 删除HDFS文件                                  |
| hadoop fs -cat           | 列出HDFS目录下的文件的内容                    |

hadoop fs -ls /   #显示hadoop根目录

#删除

hadoop fs -rmr xxxx       hadoop fs -rm-r xxxx

- 防火墙:
  - 关闭防火墙: systemctl stop firewalld
  - 查看防火墙命令: systemctl status firewalld
  - 禁用防火墙自启命令: systemctl disable firewalld
  - 启动防火墙：systemctl start firewalld.service
  - 启用防火墙自启命令: systemctl enable firewalld.service
- 退出安全模式
  - hdfs dfsadmin -safemode leave
  - 或
  - hadoop dfsadmin -safemode leave



##### HDFS的设计目标

- 适合运行在通用硬件(commodity hardware)上的分布式文件系统
- 高度容错性的系统，适合部署在廉价的机器上
- HDFS能提供高吞吐量的数据访问，非常适合大规模数据集上的应用
- 容易扩展，为用户提供性能不错的文件存储服务

##### HDFS架构

心跳机制,哨兵机制

- 1个NameNode/NN(Master) 带 DataNode/DN(Slaves) (Master-Slave结构)
- 1个文件会被拆分成多个Block
- NameNode(NN)
  - 负责客户端请求的响应
  - 负责元数据（文件的名称、副本系数、Block存放的DN）的管理
    - 元数据 MetaData 描述数据的数据
  - 监控DataNode健康状况 10分钟没有收到DataNode报告认为Datanode死掉了
- DataNode(DN),**心跳机制**
  - 存储用户的文件对应的数据块(Block)
  - 要定期向NN发送心跳信息，汇报本身及其所有的block信息，健康状况
- 分布式集群NameNode和DataNode部署在不同机器上



##### HDFS优缺点

- 优点
  - 数据冗余 硬件容错
  - 适合存储大文件
  - 处理流式数据
  - 可构建在廉价机器上
- 缺点
  - 低延迟的数据访问
  - 小文件存储



##### HDFS环境搭建

由java开发,需要 JDK环境

JDK java的开发运行环境

- 大部分的大数据框架都是用java或者scala开发的
  - scala  Java虚拟机语言
  - java -> .class->.jar   .jar文件就是java虚拟机的可执行文件
  - scala语法和java有区别  .scala -> .class ->.jar 



jps 查看java进程





##### HDFS 读写流程

client--namenode(中介)--datanodes


- 写流程
  - 客户端负责数据的拆分，拆成128MB一块的小文件
  - NameNode根据设置的副本数量，负责返回要保存的DataNode列表，如果是3副本，每一个block 返回3台DataNode的URL地址
  - DataNode 负责数据的保存，和数据的复制
    - 客户端只需要把数据和列表提交给列表中的第一台机器， DataNode之间数据复制DataNode自己完成
- 读流程
  - 客户端提交文件名给NameNode
  - NameNode返回当前文件对应哪些block,以及每一个block的所有DataNode地址
  - 客户端到地址列表中的第一台DataNode取数据




##### HDFS如何实现高可用(HA)

- 数据存储故障容错
  - 磁盘介质在存储过程中受环境或者老化影响,数据可能错乱
  - 对于存储在 DataNode 上的数据块，计算并存储校验和（CheckSum)
  - 读取数据的时候, 重新计算读取出来的数据校验和, 校验不正确抛出异常, 从其它DataNode上读取备份数据
- 磁盘故障容错
  - DataNode 监测到本机的某块磁盘损坏
  - 将该块磁盘上存储的所有 BlockID 报告给 NameNode
  - NameNode 检查这些数据块在哪些DataNode上有备份,
  - 通知相应DataNode, 将数据复制到其他服务器上
- DataNode故障容错
  - 通过心跳和NameNode保持通讯
  - 超时未发送心跳, NameNode会认为这个DataNode已经宕机
  - NameNode查找这个DataNode上有哪些数据块, 以及这些数据在其它DataNode服务器上的存储情况
  - 从其它DataNode服务器上复制数据
- NameNode故障容错
  - 主从热备 secondary namenode
  - zookeeper配合 master节点选举



##### Hadoop版本选择



#### Hadoop生态圈



##### 广义的Hadoop

- 指Hadoop生态系统，Hadoop生态系统是一个很庞大的概念，hadoop是其中最重要最基础的一个部分，生态系统中每一子系统只解决某一个特定的问题域（甚至可能更窄），不搞统一型的全能系统，而是小而精的多个小系统；





##### Hive

数据仓库,SQL Query

- sql操作 MapReduce

- Hive 由 Facebook 实现并开源，是基于 Hadoop 的一个数据仓库工具，可以将结构化的数据映射为一张数据库表，并提供 HQL(Hive SQL)查询功能，底层数据是存储在 HDFS 上。
- Hive 本质: 将 SQL 语句转换为 MapReduce 任务运行，使不熟悉 MapReduce 的用户很方便地利用 HQL 处理和计算 HDFS 上的结构化的数据,是一款基于 HDFS 的 MapReduce **计算框架**
- 主要用途：用来做离线数据分析，比直接用 MapReduce 开发效率更高。





##### Spark

分布式的计算框架基于内存

- spark core
  - 工具性组件
- spark sql
  - 离线
- spark streaming 
  - 准实时 不算是一个标准的流式计算(实时)
- spark ML (机器学习库)
- spark MLlib (机器学习库)



##### Flink

: 分布式的流式计算框架(阿里)



**Storm**: 分布式的流式计算框架 python操作storm,未提供机器学习模块

Kafka: 消息队列

Mahout:机器学习库

- 基于 java

##### **Sqoop**

数据交换框架，例如：关系型数据库与HDFS之间的数据交换

- 数据交换
- 数据导入导出

##### Hbase

nosql ,**列式数据库**,海量数据中的查询，相当于分布式文件系统中的数据库,mysql,redis为行数据库

- 适合简单的表结构,没有join关联

R:数据分析

pig：脚本语言，跟Hive类似

Oozie:工作流引擎，管理作业执行顺序

Zookeeper:用户无感知，主节点挂掉选择从节点作为主的

- 机器协调,管理员
- 保存数据一致性
- 主节点选举

Flume:日志收集框架





##### Hadoop生态系统的特点

- 开源、社区活跃
- 囊括了大数据处理的方方面面
- 成熟的生态圈



##### 方案

HDFS+ HIVE +MapReduce

HDFS＋Spark





#### 互联网大数据平台架构

- 数据采集
  - App/Web 产生的数据&日志同步到大数据系统
  - 数据库同步:Sqoop 日志同步:Flume 打点: Kafka
  - 不同数据源产生的数据质量可能差别很大
    - 数据库 也许可以直接用
    - 日志 爬虫 大量的清洗,转化处理
- 数据处理
  - 大数据存储与计算的核心
  - 数据同步后导入HDFS
  - MapReduce Hive Spark 读取数据进行计算 结果再保存到HDFS
  - MapReduce Hive Spark 离线计算, HDFS 离线存储
    - 离线计算通常针对(某一类别)全体数据, 比如 历史上所有订单
    - 离线计算特点: 数据规模大, 运行时间长
  - 流式计算
    - 淘宝双11 每秒产生订单数 监控宣传
    - Storm(毫秒) SparkStreaming(秒)
- 数据输出与展示
  - HDFS需要把数据导出交给应用程序, 让用户实时展示 ECharts
    - 淘宝卖家量子魔方
  - 给运营和决策层提供各种统计报告, 数据需要写入数据库
    - 很多运营管理人员, 上班后就会登陆后台数据系统
- 任务调度系统
  - 将上面三个部分整合起来



#### 大数据应用--数据分析

- 通过数据分析指标监控企业运营状态, 及时调整运营和产品策略,是大数据技术的关键价值之一

- 大数据平台(互联网企业)运行的绝大多数大数据计算都是关于数据分析的

  - 统计指标
  - 关联分析,
  - 汇总报告,

- 运营数据是公司管理的基础

  - 了解公司目前发展的状况
  - 数据驱动运营: 调节指标对公司进行管理

- 运营数据的获取需要大数据平台的支持

  - 埋点采集数据

  - 数据库,日志 三方采集数据

  - 对数据清洗 转换 存储

  - 利用SQL进行数据统计 汇总 分析

  - 得到需要的运营数据报告

##### 运营常用**数据指标**

  - 新增用户数 UG user growth 用户增长

    - 产品增长性的关键指标
    - 新增访问网站(新下载APP)的用户数

  - 用户留存率

    - 用户留存率 = 留存用户数 / 当期新增用户数
    - 3日留存 5日留存 7日留存

  - 活跃用户数

    - 打开使用产品的用户
    - 日活
    - 月活
    - 提升活跃是网站运营的重要目标

  - PV Page View

    - 打开产品就算活跃
    - 打开以后是否频繁操作就用PV衡量, 每次点击, 页面跳转都记一次PV

  - GMV

    - 成交总金额(Gross Merchandise Volume) 电商网站统计营业额, 反应网站应收能力的重要指标
    - GMV相关的指标: 订单量 客单价

  - **转化率**

    转化率 = 有购买行为的用户数 / 总访问用户数



#### Hive数据仓库

##### 为什么使用 Hive

- 直接使用 Hadoop MapReduce 处理数据所面临的问题：
  - 人员学习成本太高
  - MapReduce 实现复杂查询逻辑开发难度太大
- 使用 Hive
  - 操作接口采用类 SQL 语法，提供快速开发的能力
  - 避免了去写 MapReduce，减少开发人员的学习成本
  - 功能扩展很方便



##### Hive底层执行引擎

- MapReduce
- Tez
- Spark



##### Hive 与传统数据库对比

|              | Hive                              | 关系型数据库           |
| ------------ | --------------------------------- | ---------------------- |
| ANSI SQL     | 不完全支持                        | 支持                   |
| 更新         | INSERT OVERWRITE\INTO TABLE(默认) | UPDATE\INSERT\DELETE   |
| 事务         | 不支持(默认)                      | 支持                   |
| 模式         | 读模式                            | 写模式                 |
| 查询语言     | HQL                               | SQL                    |
| 数据存储     | HDFS                              | Raw Device or Local FS |
| 执行         | MapReduce                         | Executor               |
| 执行延迟     | 高                                | 低                     |
| 子查询       | 只能用在From子句中                | 完全支持               |
| 处理数据规模 | 大                                | 小                     |
| 可扩展性     | 高                                | 低                     |
| 索引         | 0.8版本后加入位图索引             | 有复杂的索引           |



##### 体系架构



##### 架构图






##### Hive HQL操作

```mysql
CREATE DATABASE test;
SHOW DATABASES;
CREATE TABLE student(classNo string, stuNo string, score int) row format delimited fields terminated by ',';
#将数据load到表中
load data local inpath '/home/hadoop/tmp/student.txt'overwrite into table student;
#查询表中的数据 跟SQL类似
hive>select * from student;
#分组查询group by和统计 count
hive>select classNo,count(score) from student where score>=60 group by classNo;


CREATE EXTERNAL TABLE student2 (classNo string, stuNo string, score int) row format delimited fields terminated by ',' location '/tmp/student';
#显示表信息
desc formatted table_name;
#删除表查看结果
drop table student;

create table employee (name string,salary bigint) partitioned by (date1 string) row format delimited fields terminated by ',' lines terminated by '\n' stored as textfile;

#-------查看表的分区-------
show partitions employee;
#-------添加分区-------
alter table employee add if not exists partition(date1='2018-12-01');
#加载数据到分区
load data local inpath '/root/tmp/employee.txt' into table employee partition(date1='2018-12-01');
#此时查看表中数据发现数据并没有变化, 需要通过hql添加分区
alter table employee add if not exists partition(date1='2018-12-04');

```

##### 动态分区

```mysql
#在写入数据时自动创建分区(包括目录结构)
#创建表
create table employee2 (name string,salary bigint) partitioned by (date1 string) row format delimited fields terminated by ',' lines terminated by '\n' stored as textfile;
#导入数据
insert into table employee2 partition(date1) select name,salary,date1 from employee;
#使用动态分区需要设置参数
set hive.exec.dynamic.partition.mode=nonstrict;
```

总结

- 利用分区表方式减少查询时需要扫描的数据量
  - 分区字段不是表中的列, 数据文件中没有对应的列
  - 分区仅仅是一个目录名
  - 查看数据时, hive会自动添加分区列
  - 支持多级分区, 多级子目录



##### Hive 函数













#### Hbase列数据库

- HBase是一个分布式的、面向列的开源数据库
- HBase是Google BigTable的开源实现
- HBase不同于一般的关系数据库, 适合非结构化数据存储



##### HBase 与 传统关系数据库的区别

|            | HBase                 | 关系型数据库              |
| ---------- | --------------------- | ------------------------- |
| 数据库大小 | PB级别                | GB ,TB                    |
| 数据类型   | Bytes                 | 丰富的数据类型            |
| 事务支持   | ACID只支持单个Row级别 | 全面的ACID支持, 对Row和表 |
| 索引       | 只支持Row-key         | 支持                      |
| 吞吐量     | 百万写入/秒           | 数千写入/秒               |









#### Spark大数据计算平台



目前，Spark已经发展成为包含众多子项目的大数据计算平台。 伯克利将Spark的整个生态系统称为伯克利数据分析栈（BDAS）。 其核心框架是Spark，同时BDAS涵盖支持结构化数据SQL查询与分析的查询引擎Spark SQL和Shark，提供机器学习功能的系统MLbase及底层的分布式机器学习库MLlib、 并行图计算框架GraphX、 流计算框架Spark Streaming、 采样近似计算查询引擎BlinkDB、 内存分布式文件系统Tachyon、 资源管理框架Mesos等子项目。 这些子项目在Spark上层提供了更高层、 更丰富的计算范式。

##### Spark生态

![img](https://images2015.cnblogs.com/blog/855959/201607/855959-20160726115216528-1598079432.png)





##### Spark特点

- speed
- ease of use
- generality
- runs everywhere



##### RDD概述

RDD（Resilient Distributed Dataset）叫做弹性分布式数据集，是Spark中最基本的数据抽象，它代表一个不可变、可分区、里面的元素可并行计算的集合.



##### Spark与Hadoop生态系统对比

##### Spark与Hadoop对比

MapReduce与Spark对比

Spark与Hadoop协作





##### Spark SQL

shark(Hive on Spark,Spark SQL)



##### SQL on Hadoop

- Hive
- impala
- presto
- drill
- Spark SQL