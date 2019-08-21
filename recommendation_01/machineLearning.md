#### 

​    



### 机器学习库



##### Machine Learning

*Libraries for Machine Learning. Also see awesome-machine-learning.*

- [H2O](https://github.com/h2oai/h2o-3) - Open Source Fast Scalable Machine Learning Platform.
- [Metrics](https://github.com/benhamner/Metrics) - Machine learning evaluation metrics.
- [NuPIC](https://github.com/numenta/nupic) - Numenta Platform for Intelligent Computing.
- [scikit-learn](http://scikit-learn.org/) - The most popular Python library for Machine Learning.
- [Spark ML](http://spark.apache.org/docs/latest/ml-guide.html) - [Apache Spark](http://spark.apache.org/)'s scalable Machine Learning library.
- [vowpal_porpoise](https://github.com/josephreisinger/vowpal_porpoise) - A lightweight Python wrapper for [Vowpal Wabbit](https://github.com/JohnLangford/vowpal_wabbit/).
- [xgboost](https://github.com/dmlc/xgboost) - A scalable, portable, and distributed gradient boosting library.



#### Numpy

- 了解Numpy运算速度上的优势
- 知道数组的属性，形状、类型
- 应用Numpy实现数组的基本操作
- 应用随机数组的创建实现正态分布应用
- 应用Numpy实现数组的逻辑运算
- 应用Numpy实现数组的统计运算
- 应用Numpy实现数组之间的运算

![img](D:/002--------------/create6@126.com/3d1079c567c84d98acf1ef1763aaf045/clipboard.png)



##### ndarray到底跟原生python列表有什么不同呢

ndarray在存储数据的时候，数据与数据的地址都是连续的，这样就给使得批量操作数组元素时速度更快。

这是因为ndarray中的所有元素的类型都是相同的，而Python列表中的元素类型是任意的，所以ndarray在存储元素时内存可以连续，而python原生list就只能通过寻址方式找到下一个元素，这虽然也导致了在通用性能方面Numpy的ndarray不及Python原生list，但在科学计算中，Numpy的ndarray就可以省掉很多循环语句，代码使用方面比Python原生list简单的多。

数据类型必须相同



![img](D:/002--------------/create6@126.com/1352afe5af36438093e06a30867fd900/clipboard.png)





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

​    	概念: 用于展示数据的变化情况的

​    	API: plt.plot(x, y)

- 2散点图: 用于分析两个变量的规律, 展示离散点分布情况

​    	API: plt.scatter(x, y)

- 3柱状图: 统计,对比,离散

​    	API: plt.bar(x, height, width, color)

​     x : x轴的标量序列

​     height: 标量或标量序列, 柱状图的高度,或者为应变量

​     width : 柱状图的宽度, 默认值0.8

​     align : 柱状图在x维度上的对齐方式, {‘center’, ‘edge’}, 可选, 默认: ‘center’

​     **kwargs :

​     color:选择柱状图的颜色

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

​     API: plt.hist(x, bins)

​        x : 数组或数组的序列, 表示要展示的数据

​        bins : 整数,序列 可选

​        如果是整数就是柱状体的个数

​        如果序列就是每个柱状体的边缘值, 左开右闭.

```python
x2 = np.random.normal(loc=2,scale=4,size=100000)
#loc均值,scale 标准差
#画布
plt.figure(figsize=(20,8),dpi=100)
plt.hist(x2,bins=1000)
plt.show()
```



- 5饼状图: 占比

​    API: plt.pie(x, labels, autopct, colors)

​       x:数量，自动算百分比

​       labels:每部分名称

​       autopct:占比显示指定  '%.2f%%'

​       colors:每部分颜色

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

![img](data:image/png;base64,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)

##### countplot 计数直方图
countplot 故名思意，是“计数图”的意思，可将它认为一种应用到分类变量的直方图，也可认为它是用以比较类别间计数差，调用 count 函数的 barplot；

countplot 参数和 barplot 基本差不多，可以对比着记忆，有一点不同的是 countplot 中不能同时输入 x 和 y ，且 countplot 没有误差棒。



```python
# 首先绘制玩家杀敌数的条形图
plt.figure(figsize=(10,4))
sns.countplot(data=train, x=train['kills']).set_title('Kills')
plt.show()
```

![img](data:image/png;base64,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)





#### Sklearn

- Python语言的机器学习工具
- Scikit-learn包括许多知名的机器学习算法的实现
- Scikit-learn文档完善，容易上手，丰富的API
- 目前稳定版本0.19.1

![image-20190225170704470](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/sklearn%E5%8C%85%E5%90%AB%E5%86%85%E5%AE%B9.png)



##### classification分类

常用的分类：线性、决策树、SVM、KNN，朴素贝叶斯；集成分类：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees





##### regressor 回归

常用的回归：线性、决策树、SVM、KNN ；集成回归：随机森林、Adaboost、GradientBoosting、Bagging、ExtraTrees

#from sklearn.linear_model import LinearRegression,SGDRegressor 
#线性回归模型与梯度下降模型(随机梯度下降算法)

- Ridge regression
- Logistic regression
- Ordinary least squares 
- Bayesian linear regression w/ conjugate priors 贝叶斯定理
  - Unknown mean, known variance (Gaussian prior)
  - Unknown mean, unknown variance (Normal-Gamma / Normal-Inverse-Wishart prior)





##### clustering 聚类

常用聚类：k均值（K-means）、层次聚类（Hierarchical clustering）、DBSCAN





##### Dimensionality reduction 降维

常用降维：LinearDiscriminantAnalysis、PCA





##### Model selection 模型选择      

#from sklearn.model_selection import train_test_split,GridSearchCV 数据集分割,网格搜索与交叉验证

- 3.1 交叉验证：评估估算器性能
  - 3.1.1 计算交叉验证的指标
    - [3.1.1.1 cross_validate函数和多个度量评估](https://scikit-learn.org/stable/modules/cross_validation.html#the-cross-validate-function-and-multiple-metric-evaluation)
    - [3.1.1.2 通过交叉验证获得预测](https://scikit-learn.org/stable/modules/cross_validation.html#obtaining-predictions-by-cross-validation)
  - 3.1.2 交叉验证迭代器
    - 3.1.2.1 iid数据的交叉验证迭代器
      - [3.1.2.1.1 K-倍](https://scikit-learn.org/stable/modules/cross_validation.html#k-fold)
      - [3.1.2.1.2 重复K-Fold](https://scikit-learn.org/stable/modules/cross_validation.html#repeated-k-fold)
      - [3.1.2.1.3 离开一个人（LOO）](https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-out-loo)
      - [3.1.2.1.4 离开P Out（LPO）](https://scikit-learn.org/stable/modules/cross_validation.html#leave-p-out-lpo)
      - [3.1.2.1.5 随机排列交叉验证又名Shuffle＆Split](https://scikit-learn.org/stable/modules/cross_validation.html#random-permutations-cross-validation-a-k-a-shuffle-split)
    - 3.1.2.2 交叉验证迭代器，基于类标签进行分层 
      - [3.1.2.2.1 分层k倍](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold)
      - [3.1.2.2.2 分层随机分裂](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-shuffle-split)
    - 3.1.2.3 分组数据的交叉验证迭代器 
      - [3.1.2.3.1 组k倍](https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold)
      - [3.1.2.3.2 离开一个小组](https://scikit-learn.org/stable/modules/cross_validation.html#leave-one-group-out)
      - [3.1.2.3.3 离开P组](https://scikit-learn.org/stable/modules/cross_validation.html#leave-p-groups-out)
      - [3.1.2.3.4 Group Shuffle Split](https://scikit-learn.org/stable/modules/cross_validation.html#group-shuffle-split)
    - [3.1.2.4 预定义的折叠 - 拆分/验证集](https://scikit-learn.org/stable/modules/cross_validation.html#predefined-fold-splits-validation-sets)
    - 3.1.2.5 交叉验证时间序列数据
      - [3.1.2.5.1 时间序列分裂](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split)
  - [3.1.3 关于改组的说明](https://scikit-learn.org/stable/modules/cross_validation.html#a-note-on-shuffling)
  - [3.1.4 交叉验证和模型选择](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-and-model-selection)
- 3.2 调整估计器的超参数
  - [3.2.1 穷举网格搜索](https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search)
  - [3.2.2 随机参数优化](https://scikit-learn.org/stable/modules/grid_search.html#randomized-parameter-optimization)
  - 3.2.3 参数搜索提示
    - [3.2.3.1 指定客观指标](https://scikit-learn.org/stable/modules/grid_search.html#specifying-an-objective-metric)
    - [3.2.3.2 指定多个评估指标](https://scikit-learn.org/stable/modules/grid_search.html#specifying-multiple-metrics-for-evaluation)
    - [3.2.3.3 复合估计器和参数空间](https://scikit-learn.org/stable/modules/grid_search.html#composite-estimators-and-parameter-spaces)
    - [3.2.3.4 模型选择：开发和评估](https://scikit-learn.org/stable/modules/grid_search.html#model-selection-development-and-evaluation)
    - [3.2.3.5 排比](https://scikit-learn.org/stable/modules/grid_search.html#parallelism)
    - [3.2.3.6 对失败的坚定性](https://scikit-learn.org/stable/modules/grid_search.html#robustness-to-failure)
  - 3.2.4 强力参数搜索的替代方案
    - 3.2.4.1 模型特定的交叉验证
      - [3.2.4.1.1 `sklearn.linear_model`.ElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html)
      - [3.2.4.1.2 `sklearn.linear_model`.LarsCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LarsCV.html)
      - 3.2.4.1.3 `sklearn.linear_model`.LassoCV
        - [3.2.4.1.3.1 使用示例`sklearn.linear_model.LassoCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#examples-using-sklearn-linear-model-lassocv)
      - 3.2.4.1.4 `sklearn.linear_model`.LassoLarsCV
        - [3.2.4.1.4.1 使用示例`sklearn.linear_model.LassoLarsCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsCV.html#examples-using-sklearn-linear-model-lassolarscv)
      - [3.2.4.1.5 `sklearn.linear_model`.LogisticRegressionCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html)
      - [3.2.4.1.6 `sklearn.linear_model`.MultiTaskElasticNetCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskElasticNetCV.html)
      - [3.2.4.1.7 `sklearn.linear_model`.MultiTaskLassoCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.MultiTaskLassoCV.html)
      - 3.2.4.1.8 `sklearn.linear_model`.OrthogonalMatchingPursuitCV
        - [3.2.4.1.8.1 使用示例`sklearn.linear_model.OrthogonalMatchingPursuitCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.OrthogonalMatchingPursuitCV.html#examples-using-sklearn-linear-model-orthogonalmatchingpursuitcv)
      - 3.2.4.1.9 `sklearn.linear_model`.RidgeCV
        - [3.2.4.1.9.1 使用示例`sklearn.linear_model.RidgeCV`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html#examples-using-sklearn-linear-model-ridgecv)
      - [3.2.4.1.10 `sklearn.linear_model`.RidgeClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeClassifierCV.html)
    - 3.2.4.2 信息标准
      - 3.2.4.2.1 `sklearn.linear_model`.LassoLarsIC
        - [3.2.4.2.1.1 使用示例`sklearn.linear_model.LassoLarsIC`](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLarsIC.html#examples-using-sklearn-linear-model-lassolarsic)
    - 3.2.4.3 Out of Bag Estimates
      - 3.2.4.3.1 `sklearn.ensemble`.RandomForestClassifier
        - [3.2.4.3.1.1 使用示例`sklearn.ensemble.RandomForestClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#examples-using-sklearn-ensemble-randomforestclassifier)
      - 3.2.4.3.2 `sklearn.ensemble`.RandomForestRegressor
        - [3.2.4.3.2.1 使用示例`sklearn.ensemble.RandomForestRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#examples-using-sklearn-ensemble-randomforestregressor)
      - 3.2.4.3.3 `sklearn.ensemble`.ExtraTreesClassifier
        - [3.2.4.3.3.1 使用示例`sklearn.ensemble.ExtraTreesClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#examples-using-sklearn-ensemble-extratreesclassifier)
      - 3.2.4.3.4 `sklearn.ensemble`.ExtraTreesRegressor
        - [3.2.4.3.4.1 使用示例`sklearn.ensemble.ExtraTreesRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#examples-using-sklearn-ensemble-extratreesregressor)
      - 3.2.4.3.5 `sklearn.ensemble`.GradientBoostingClassifier
        - [3.2.4.3.5.1 使用示例`sklearn.ensemble.GradientBoostingClassifier`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#examples-using-sklearn-ensemble-gradientboostingclassifier)
      - 3.2.4.3.6 `sklearn.ensemble`.GradientBoostingRegressor
        - [3.2.4.3.6.1 使用示例`sklearn.ensemble.GradientBoostingRegressor`](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#examples-using-sklearn-ensemble-gradientboostingregressor)
- 3.3 模型评估：量化预测的质量
  - 3.3.1 该`scoring`参数：定义模型评估规则
    - [3.3.1.1 常见情况：预定义值](https://scikit-learn.org/stable/modules/model_evaluation.html#common-cases-predefined-values)
    - [3.3.1.2 从度量函数定义评分策略](https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions)
    - [3.3.1.3 实现自己的评分对象](https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object)
    - [3.3.1.4 使用多指标评估](https://scikit-learn.org/stable/modules/model_evaluation.html#using-multiple-metric-evaluation)
  - 3.3.2 分类指标
    - [3.3.2.1 从二进制到多类和多标签](https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel)
    - [3.3.2.2 准确度得分](https://scikit-learn.org/stable/modules/model_evaluation.html#accuracy-score)
    - [3.3.2.3 平衡准确度得分](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score)
    - [3.3.2.4 科恩的卡帕](https://scikit-learn.org/stable/modules/model_evaluation.html#cohen-s-kappa)
    - [3.3.2.5 混淆矩阵](https://scikit-learn.org/stable/modules/model_evaluation.html#confusion-matrix)
    - [3.3.2.6 分类报告](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-report)
    - [3.3.2.7 汉明失利](https://scikit-learn.org/stable/modules/model_evaluation.html#hamming-loss)
    - 3.3.2.8 精确度，召回率和F度量
      - [3.3.2.8.1 二进制分类](https://scikit-learn.org/stable/modules/model_evaluation.html#binary-classification)
      - [3.3.2.8.2 多类和多标签分类](https://scikit-learn.org/stable/modules/model_evaluation.html#multiclass-and-multilabel-classification)
    - [3.3.2.9 Jaccard相似系数得分](https://scikit-learn.org/stable/modules/model_evaluation.html#jaccard-similarity-coefficient-score)
    - [3.3.2.10 铰链损失](https://scikit-learn.org/stable/modules/model_evaluation.html#hinge-loss)
    - [3.3.2.11 记录丢失](https://scikit-learn.org/stable/modules/model_evaluation.html#log-loss)
    - [3.3.2.12 马修斯相关系数](https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-correlation-coefficient)
    - [3.3.2.13 多标签混淆矩阵](https://scikit-learn.org/stable/modules/model_evaluation.html#multi-label-confusion-matrix)
    - [3.3.2.14 接收器工作特性（ROC）](https://scikit-learn.org/stable/modules/model_evaluation.html#receiver-operating-characteristic-roc)
    - [3.3.2.15 零损失](https://scikit-learn.org/stable/modules/model_evaluation.html#zero-one-loss)
    - [3.3.2.16 布里尔得分亏损](https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss)
  - 3.3.3 多标签排名指标
    - [3.3.3.1 覆盖率错误](https://scikit-learn.org/stable/modules/model_evaluation.html#coverage-error)
    - [3.3.3.2 标签排名平均精度](https://scikit-learn.org/stable/modules/model_evaluation.html#label-ranking-average-precision)
    - [3.3.3.3 排名亏损](https://scikit-learn.org/stable/modules/model_evaluation.html#ranking-loss)
  - 3.3.4 回归指标
    - [3.3.4.1 解释方差分数](https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score)
    - [3.3.4.2 最大错误](https://scikit-learn.org/stable/modules/model_evaluation.html#max-error)
    - [3.3.4.3 平均绝对误差](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error)
    - [3.3.4.4 均方误差](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-error)
    - [3.3.4.5 均方对数误差平均值](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-squared-logarithmic-error)
    - [3.3.4.6 中位数绝对误差](https://scikit-learn.org/stable/modules/model_evaluation.html#median-absolute-error)
    - [3.3.4.7 R²分数，决定系数](https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score-the-coefficient-of-determination)
  - [3.3.5 群集指标](https://scikit-learn.org/stable/modules/model_evaluation.html#clustering-metrics)
  - [3.3.6 假人估计](https://scikit-learn.org/stable/modules/model_evaluation.html#dummy-estimators)
- 3.4 模型持久性
  - [3.4.1 持久性的例子](https://scikit-learn.org/stable/modules/model_persistence.html#persistence-example)
  - [3.4.2 安全性和可维护性限制](https://scikit-learn.org/stable/modules/model_persistence.html#security-maintainability-limitations)
- 3.5 验证曲线：绘制分数以评估模型
  - [3.5.1 验证曲线](https://scikit-learn.org/stable/modules/learning_curve.html#validation-curve)
  - [3.5.2 学习曲线](https://scikit-learn.org/stable/modules/learning_curve.html#learning-curve)



##### Preprocessing (特征)预处理   

#from sklearn.preprocessing import StandardScaler,MinMaxScaler 标准化与归一化

- Discrete Fourier transform (1D signals)   离散傅里叶变换
- Discrete cosine transform (type-II) (1D signals) 离散余弦变换
- Bilinear interpolation (2D signals)  双线性插值
- Nearest neighbor interpolation (1D and 2D signals) **最近邻插值**
- Autocorrelation (1D signals)  [数] 自相关（作用）
- Signal windowing  信号窗口
- Text tokenization  文本标记
- Feature hashing 特征哈希
- Feature standardization 特征标准化
- One-hot encoding / decoding 热编码,解码
  - from sklearn.feature_extraction import DictVectorizer
- Huffman coding / decoding 霍夫曼编码,解码
- Term frequency-inverse document frequency encoding  项频率逆文档频率编码
- MFCC encoding  MFCC编码



5.3 预处理数据

- 5.3.1 标准化，或平均删除和方差缩放
  - [5.3.1.1 将功能扩展到范围](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-features-to-a-range)
  - [5.3.1.2 缩放稀疏数据](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-sparse-data)
  - [5.3.1.3 使用异常值缩放数据](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-data-with-outliers)
  - [5.3.1.4 居中核矩阵](https://scikit-learn.org/stable/modules/preprocessing.html#centering-kernel-matrices)
- 5.3.2 非线性变换
  - [5.3.2.1 映射到统一分布](https://scikit-learn.org/stable/modules/preprocessing.html#mapping-to-a-uniform-distribution)
  - [5.3.2.2 映射到高斯分布](https://scikit-learn.org/stable/modules/preprocessing.html#mapping-to-a-gaussian-distribution)
- [5.3.3 正常化](https://scikit-learn.org/stable/modules/preprocessing.html#normalization)
- [5.3.4 编码分类功能](https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features)
- 5.3.5 离散
  - [5.3.5.1 K-bin离散化](https://scikit-learn.org/stable/modules/preprocessing.html#k-bins-discretization)
  - [5.3.5.2 特征二值化](https://scikit-learn.org/stable/modules/preprocessing.html#feature-binarization)
- [5.3.6 估算缺失值](https://scikit-learn.org/stable/modules/preprocessing.html#imputation-of-missing-values)
- [5.3.7 生成多项式特征](https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features)
- [5.3.8 定制变压器](https://scikit-learn.org/stable/modules/preprocessing.html#custom-transformers)



#####  sklearn模型的保存和加载API

- from sklearn.externals import joblib
  - 保存：joblib.dump(estimator, 'test.pkl')
  - 加载：estimator = joblib.load('test.pkl')
  - 注意:  文件名的后缀是 `pkl



![基于随机森林的房屋价格预测](D:\003_IT\download\Tech Fin研习社机器学习-风控\评分卡项目\基于随机森林算法的房屋价格预测模型\基于随机森林的房屋价格预测.png)

![scikit-learnç®æ³éæ©è·¯å¾å¾](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/scikit-learn%E7%AE%97%E6%B3%95%E9%80%89%E6%8B%A9%E8%B7%AF%E5%BE%84%E5%9B%BE.png)



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

归一化

标准化

离散化(eg.sigmoid二分类,)



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

​      输入数据: 特征 和 目标值
​      算法: 
​        目标值连续: 回归问题		线性回归	 	  	
​        目标值离散: 分类问题		KNN

#### 无监督学习

​      数据数据: 只有特征值没有目标值
​      算法: 聚类

#### 半监督学习

​      输入数据: 特征 + 目标值(部分)

#### 强化学习

​    	概念:  智能体(Agent)不断与环境进行交互, 通过试错得到最佳策略. 

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

![Ã¥Â½âÃ¤Â¸â¬Ã¥ÅâÃ¥â¦Â¬Ã¥Â¼Â](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/%E5%BD%92%E4%B8%80%E5%8C%96%E5%85%AC%E5%BC%8F.png)

##### 标准化表达式:

![Ã¦ â¡Ã¥â¡â Ã¥ÅâÃ¥â¦Â¬Ã¥Â¼Â](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/%E6%A0%87%E5%87%86%E5%8C%96%E5%85%AC%E5%BC%8F.png)

![1564997318681](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1564997318681.png)

- 通常使用标准化
- 为什么异常值对归一化的影响很大，请简要说明

由其计算公式可知,一组数据中的极值对结果有决定作用,当一组数据中有异常值,通常是偏大或者偏小,既会归一化结果.

﻿最大值最小值是变化的，另外，最大值与最小值非常容易受异常点影响，**所以这种方法鲁棒性较差，只适合传统精确小数据场景。**



#### 交叉验证

- 目的: 为了提高模型训练结果可信度.
- 防止特殊数据集中在一个区域



![1565058909701](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565058909701.png)

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





![img](D:/002--------------/create6@126.com/05ff413c601d4874a450784ebbc471f7/clipboard.png)



![img](D:/002--------------/create6@126.com/46ddc452fe5e4bedacfc954aff52d91a/clipboard.png)



![img](D:/002--------------/create6@126.com/2b9434d7e64e4316aed2d2a9a218b712/clipboard.png)

#### 似然函数,对数似然

- 让预测值越接近真实值(极大似然),让似然函数越大越好,减号右边的式子越小越好,得到最小二乘法

![img](D:/002--------------/create6@126.com/c4406e4640404604a50b15fa06de0351/clipboard.png)



#### 最小二乘法

![img](D:/002--------------/create6@126.com/23e042e841bb446a90942f5666d624fb/clipboard.png)



![img](file:///D:/002--------------/create6@126.com/9f6e6a58f59d496398ab2fbad0ab66af/clipboard.png)

#### 评估方法

![img](file:///D:/002--------------/create6@126.com/ecdb7f514b984249a9ddd957c79f62f9/clipboard.png)

#### 通用公式cz

线性回归(Linear regression)是利用**回归方程(函数)**对**一个或多个自变量(特征值)和因变量(目标值)之间**关系进行建模的一种分析方式。

![1565076966571](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565076966571.png)

####  损失函数

![1565080170451](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565080170451.png)

- yi为第i个训练样本的真实值
- h(xi)为第i个训练样本特征值组合预测函数
- 又称最小二乘法

如何去减少这个损失，使我们预测的更加准确些？既然存在了这个损失，我们一直说机器学习有自动学习的功能，在线性回归这里更是能够体现。这里可以通过一些优化方法去优化（其实是数学当中的求导功能）回归的总损失



#### 优化算法

- 如何去求模型当中的θ，使得损失最小？（目的是找到最小损失对应的θ值）



  ##### 线性回归经常使用的两种优化算法



  ![1565084343826](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565084343826.png)

- [x] 1.正规方程: 根据样本数据直接计算一个最好的模型系数和偏置(**计算量大**)

  ![1565080686372](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565080686372.png)

![1565081068506](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565081068506.png)

​		(1)中 要消掉X,X不一定是方阵,先转置,再乘以X的逆来消除X



- [x] 2.梯度下降: 从一个任意的模型系数和偏置开始, 一步一步进行优化, 最终得到一个最好的模型系数和偏置.	





#### 梯度下降!!

![1564796112880](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1564796112880.png)

##### 用小的学习率,用大的迭代率

![1564796565762](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1564796565762.png)

- 单变量

![1565081845986](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565081845986.png)

![1565081872217](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565081872217.png)

- **多变量**

  - 我们假设有一个目标函数 ：:J(θ) = θ12 + θ22

    现在要通过梯度下降法计算这个函数的最小值。我们通过观察就能发现最小值其实就是 (0，0)点。但是接下 来，我们会从梯度下降算法开始一步步计算到这个最小值! 我们假设初始的起点为: θ0 = (1, 3)

    初始的学习率为:α = 0.1

    函数的梯度为:▽:J(θ) =< 2θ1 ,2θ2>



![snipaste20190806_171004](C:\Users\struggle6\Desktop\printscreen\snipaste20190806_171004.png)

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

（1**）FG方法由于它每轮更新都要使用全体数据集，故花费的时间成本最多，内存存储最大。**

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

![1565245111988](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565245111988.png)

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



![1564792468000](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1564792468000.png)





#### Sigmoid函数: 值-->概率

![1564792176079](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1564792176079.png)

![1564793148954](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1564793148954.png)

![1564798139615](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1564798139615.png)
![1565057194299](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565057194299.png)

 

#### 损失

逻辑回归的损失，称之为对数似然损失，公式如下：

分开类别：
![1565250477320](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565250477320.png)

怎么理解单个的式子呢？这个要根据log的函数图像来理解

![image-20190221142055367](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/%E5%AF%B9%E6%95%B0%E4%BC%BC%E7%84%B6%E6%8D%9F%E5%A4%B1%E5%87%BD%E6%95%B0.png)

##### 损失函数

![1565250496231](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565250496231.png)

看到这个式子，其实跟我们讲的信息熵类似。

![1565250451959](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565250451959.png)

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



![image-20190321103930761](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/confusion_matrix1.png)



- 召回率：真实为正例的样本中预测结果为正例的比例（查得全，对正样本的区分能力）

$$
Recall = \frac{TP}{TP + FN}
$$



![image-20190321103947092](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/confusion_matrix2.png)

TPR 即召回率 recall
FPR可以理解为错误率,此参数越低,损失越低

![1565953526777](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565953526777.png)

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

```python
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




![1565159747738](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565159747738.png)

![1565160215292](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565160215292.png)

```python
 >>> -(2/5)*math.log(2/5,2)-(3/5)*math.log(3/5,2)
0.9709505944546686
```

![1565161359979](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565161359979.png)



![1565161708108](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565161708108.png)

![1565161982924](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565161982924.png)

#### 决策树便于可视化展示(绘图)

![1565162471116](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565162471116.png)

![1565165493433](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565165493433.png)

#### 信息增益率
![1565261723704](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565261723704.png)

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



![1565229276235](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565229276235.png)

![1565229495013](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565229495013.png)

![1565231058136](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565231058136.png)

![1565231641853](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565231641853.png)



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

![1565233388641](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565233388641.png)

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



![1565187726469](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565187726469.png)



![1565186597309](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565186597309.png)
#### 随机森林

![1565186855375](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565186855375.png)

![基于随机森林的房屋价格预测](D:\003_IT\download\Tech Fin研习社机器学习-风控\评分卡项目\基于随机森林算法的房屋价格预测模型\基于随机森林的房屋价格预测.png)


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




![1565187455284](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565187455284.png)


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



![1565188017948](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565188017948.png)

#### Xgboost

Xgboost中国人建立,有Python库

**XGBoost= 二阶泰勒展开+boosting+决策树+正则化**

![1565239189929](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565239189929.png)

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



![1565244076756](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565244076756.png)

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

![1565229743237](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565229743237.png)

![1565230267564](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565230267564.png)

![1565230299055](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565230299055.png)

![1565230895170](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565230895170.png)

#### 核心(优化目标)

##### 找到一条线(w,b),使得离该线最近的点(雷区)能够最远

![1565236197913](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565236197913.png)









### 贝叶斯





![1565227240912](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565227240912.png)

![1565227471427](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565227471427.png)













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

![1565310284069](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565310284069.png)

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

![scikit-learnç®æ³éæ©è·¯å¾å¾](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/scikit-learn%E7%AE%97%E6%B3%95%E9%80%89%E6%8B%A9%E8%B7%AF%E5%BE%84%E5%9B%BE.png)



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



![image-20190321103930761](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/confusion_matrix1.png)



- 召回率：真实为正例的样本中预测结果为正例的比例（查得全，对正样本的区分能力）

$$
Recall = \frac{TP}{TP + FN}
$$



![image-20190321103947092](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/confusion_matrix2.png)

TPR 即召回率 recall
FPR可以理解为错误率,此参数越低,损失越低

![1565953526777](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565953526777.png)





#### 回归模型评估

- ##### 均方根误差（Root Mean Squared Error，RMSE）

  - RMSE是一个衡量回归模型误差率的常用公式。 然而，它仅能比较误差是相同单位的模型。

    ![image-20190312193846308](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E5%BA%93/images/%E5%9D%87%E6%96%B9%E6%A0%B9%E8%AF%AF%E5%B7%AE.png)

- ##### 相对平方误差（Relative Squared Error，RSE）

  - 与RMSE不同，RSE可以比较误差是不同单位的模型。

    ![image-20190312194839069](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E5%BA%93/images/%E7%9B%B8%E5%AF%B9%E5%B9%B3%E6%96%B9%E8%AF%AF%E5%B7%AE.png)

  - 其中

  ![åå¼](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E5%BA%93/images/a%E7%9C%9F%E5%AE%9E%E5%80%BC%E5%9D%87%E5%80%BC.png)





- ##### 平均绝对误差（Mean Absolute Error，MAE)

  - MAE与原始数据单位相同， 它仅能比较误差是相同单位的模型。量级近似与RMSE，但是误差值相对小一些。

    ![image-20190312194923850](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E5%BA%93/images/%E5%B9%B3%E5%9D%87%E7%BB%9D%E5%AF%B9%E8%AF%AF%E5%B7%AE.png)

- ##### 相对绝对误差（Relative Absolute Error，RAE)

  - 与RSE不同，RAE可以比较误差是不同单位的模型。

    ![image-20190312195006252](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E5%BA%93/images/%E7%9B%B8%E5%AF%B9%E7%BB%9D%E5%AF%B9%E8%AF%AF%E5%B7%AE.png)

- ##### 决定系数 (Coefficient of Determination)

  - 决定系数 (**R2**)回归模型汇总了回归模型的解释度，由平方和术语计算而得。

    ![image-20190312202620606](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E5%BA%93/images/%E5%86%B3%E5%AE%9A%E7%B3%BB%E6%95%B0.png)

  - R2描述了回归模型所解释的因变量方差在总方差中的比例。R2很大，即自变量和因变量之间存在线性关系，如果回归模型是“完美的”，SSE为零，则R2为1。R2小，则自变量和因变量之间存在线性关系的证据不令人信服。如果回归模型完全失败，SSE等于SST，没有方差可被回归解释，则R2为零。

    - 注:

      - SSE: The sum of squares due to error(误差平方和)

        ![image-20190312202620606](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E5%BA%93/images/SSE.png)

      - SST: Total sum of squares(原始数据和均值之差的平方和)

        ![image-20190312202620606](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%A7%91%E5%AD%A6%E8%AE%A1%E7%AE%97%E5%BA%93/images/SST.png)

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

![img](D:/002--------------/create6@126.com/e26760efa2f545b282fd011f2274aeaa/clipboard.png)

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

![snipaste20190807_091622](C:\Users\struggle6\Desktop\printscreen\snipaste20190807_091622.png)



![1565141715738](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565141715738.png)



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

![img](D:/002--------------/create6@126.com/48bf05ba42c84477a4157d0f699ecb88/clipboard.png)

![img](D:/002--------------/create6@126.com/bd925116501141548c8feb932db269d7/clipboard.png)





### 概率:



![img](D:/002--------------/create6@126.com/9b0dfe6ecabb463c9c12373cfde77de2/clipboard.png)



![img](D:/002--------------/create6@126.com/32430270bc554428b412d91df38f4b3f/clipboard.png)



![img](D:/002--------------/create6@126.com/129093fa96b9417c80be3fdd921c8f63/clipboard.png)



![img](D:/002--------------/create6@126.com/18bf5ca89c5d497e88ac53a2664686e0/clipboard.png)



![img](D:/002--------------/create6@126.com/680f5e87aeb84aaa8306e3f146def894/clipboard.png)



![img](D:/002--------------/create6@126.com/89ec6b7058834a67b4d20d19c6cab436/clipboard.png)





![img](D:/002--------------/create6@126.com/62bdcede7924433f8a26c94ab6dd56f2/clipboard.png)



![img](D:/002--------------/create6@126.com/cffa937ab69d429da3df0ae8369a3f5b/clipboard.png)





### 求导



![img](D:/002--------------/create6@126.com/2a1b2d5ca8d9493db326e3b8324b0ccf/clipboard.png)



#### 导数的四则运算

![image-20190319114321271](file:///G:/python%E5%AD%A6%E4%B9%A0/%E5%BD%92%E6%A1%A3-%E8%AF%BE%E4%BB%B6-%E8%A7%86%E9%A2%91/%E8%AF%BE%E4%BB%B6/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0_%E7%AE%97%E6%B3%95%E7%AF%87/images/%E6%B1%82%E5%AF%BC%E5%9B%9B%E5%88%99%E8%BF%90%E7%AE%97.png)



![img](D:/002--------------/create6@126.com/7bcf4b3bca634e42bb489a34b476feb7/clipboard.png)





![img](D:/002--------------/create6@126.com/d594799d6fd94c449e6f0013743a0fdf/clipboard.png)



![img](D:/002--------------/create6@126.com/331f6a0f47024a7ba984558d3c2b13b1/clipboard.png)



![img](D:/002--------------/create6@126.com/4cf4bcd3f7ce425a82e0d9852129f47f/clipboard.png)



![img](D:/002--------------/create6@126.com/a5be653f4b6746f791d1d97493a08806/clipboard.png)



![img](D:/002--------------/create6@126.com/0fd57570e29b40f38e8a36208d1d369a/clipboard.png)

6-6-1

![img](D:/002--------------/create6@126.com/6f44b13d1ed74ffeba8c00b282fdeac6/clipboard.png)

6-6-2

![img](D:/002--------------/create6@126.com/99c4389befdd41ec87f30f25f3a15b50/clipboard.png)





### 求左右极限:

![img](D:/002--------------/create6@126.com/a62db14ac73c4573aafb54c9d99be0ff/clipboard.png)



![img](D:/002--------------/create6@126.com/7f2c89df5d9c493480af5f5639ce2f29/clipboard.png)



![img](D:/002--------------/create6@126.com/718ea90aadab453a9efbde4d34087889/clipboard.png)



![img](D:/002--------------/create6@126.com/7a3c53be46ac41cf9d5f50b7a891b034/clipboard.png)



### 极限:





![img](D:/002--------------/create6@126.com/d1ecfb4eaba24f88a543dfb71ed8a59e/clipboard.png)



![img](D:/002--------------/create6@126.com/0d0069edd6ac43f4b1cd4bfe3f842a29/clipboard.png)

4-4-1 直接使用

![img](D:/002--------------/create6@126.com/d4a389a195ec4199a2dbbda9f7acf3d2/clipboard.png)

4-4-2 变指数

![img](D:/002--------------/create6@126.com/436cbd4e1ff94d6ba4bc25113290b1ad/clipboard.png)

4-4-3 变底数

![img](D:/002--------------/create6@126.com/0e1dfbf72e27405f96d4893e473655cb/clipboard.png)

4-4-4 变指数与底数:xxx











#### 微软机器学习模拟

![img](D:/002--------------/create6@126.com/3da537886f2444a990022422115b7bd1/clipboard.png)





### 深度学习

![1565315523351](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565315523351.png)



当数据量变大时,深度学习的效果更好



#### 常规思路

##### 1.收集数据并给定标签

##### 2.训练一个分类器

##### 3.测试评估



















### 神经网络

#### 结构

![1565945424918](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565945424918.png)


![1565945602337](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565945602337.png)

![1565946712794](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565946712794.png)



激活函数(非线性)

![1565945624731](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565945624731.png)

##### 神经网络首选ReLU激活函数



![1565946071322](C:\Users\struggle6\AppData\Roaming\Typora\typora-user-images\1565946071322.png)



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



