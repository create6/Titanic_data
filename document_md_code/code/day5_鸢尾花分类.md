
### 鸢尾花类别预测


```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
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
estimator = KNeighborsClassifier(n_neighbors=5)
#学习:训练集特征值,训练集目标值
estimator.fit(x_train,y_train)
#5,模型评估
#使用模型对测试集进行预测,参数为测试集的特征值,返回为预测目标值
y_pre =estimator.predict(x_test)
print(y_pre)
print('y_pre==y_test:\n',y_pre == y_test)
#准确率
score =estimator.score(x_test,y_test)
print('准确率:',score)#random_state=22,score= 0.911111,random_state=8,score还是0.911111
```

    [0 0 0 2 1 0 0 2 2 1 1 0 1 1 1 2 2 2 2 2 1 0 1 1 1 0 2 0 0 2 0 0 0 2 1 1 2
     1 0 1 1 1 1 1 2]
    y_pre==y_test:
     [ True  True  True  True  True  True  True  True  True  True  True  True
     False  True False  True  True  True False  True  True  True  True  True
      True  True  True  True  True  True  True  True  True  True  True  True
      True  True  True  True  True False  True  True  True]
    准确率: 0.9111111111111111
    

#### 交叉验证


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

    准确率: 0.9111111111111111
    交叉验证最好分数: 0.9714285714285714
    交叉验证最好的模型: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=7, p=2,
               weights='uniform')
    交叉验证的结果:
     {'mean_fit_time': array([0.0010004 , 0.00080023, 0.00080161, 0.00099916]), 'std_fit_time': array([1.39020727e-06, 4.00116006e-04, 4.00825225e-04, 7.44843452e-07]), 'mean_score_time': array([0.00199838, 0.00159802, 0.00199738, 0.00159922]), 'std_score_time': array([1.09406056e-03, 4.88928337e-04, 3.78657946e-06, 4.89512341e-04]), 'param_n_neighbors': masked_array(data=[3, 5, 7, 9],
                 mask=[False, False, False, False],
           fill_value='?',
                dtype=object), 'params': [{'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 7}, {'n_neighbors': 9}], 'split0_test_score': array([0.90909091, 0.90909091, 0.90909091, 0.95454545]), 'split1_test_score': array([0.95238095, 1.        , 1.        , 1.        ]), 'split2_test_score': array([1.        , 1.        , 1.        , 0.95238095]), 'split3_test_score': array([1., 1., 1., 1.]), 'split4_test_score': array([0.9 , 0.9 , 0.95, 0.95]), 'mean_test_score': array([0.95238095, 0.96190476, 0.97142857, 0.97142857]), 'std_test_score': array([0.04268846, 0.04674523, 0.03730235, 0.02337261]), 'rank_test_score': array([4, 3, 1, 1]), 'split0_train_score': array([0.97590361, 0.96385542, 0.97590361, 0.97590361]), 'split1_train_score': array([0.96428571, 0.96428571, 0.96428571, 0.95238095]), 'split2_train_score': array([0.97619048, 0.97619048, 0.98809524, 0.96428571]), 'split3_train_score': array([0.97619048, 0.95238095, 0.97619048, 0.96428571]), 'split4_train_score': array([1.        , 1.        , 1.        , 0.98823529]), 'mean_train_score': array([0.97851406, 0.97134251, 0.98089501, 0.96901826]), 'std_train_score': array([0.01167651, 0.01618734, 0.01216355, 0.01215151])}
    


```python

```


```python

```


```python

```


```python

```


```python

```
