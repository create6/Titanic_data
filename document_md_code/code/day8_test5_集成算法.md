
### 分别使用决策树, 随机森林, KNN算法, 并使用网格搜索和交叉验证对参数进行调优; 得到一个最佳的算法模型.


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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
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

    False
    


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
estimator = DecisionTreeClassifier()
estimator.fit(x_train,y_train)


```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=1, min_samples_split=2,
                min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                splitter='best')




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

    准确率: 0.7288888888888889
                  precision    recall  f1-score   support
    
               0       0.83      0.82      0.82      6995
               1       0.40      0.42      0.41      2005
    
       micro avg       0.73      0.73      0.73      9000
       macro avg       0.61      0.62      0.62      9000
    weighted avg       0.73      0.73      0.73      9000
    
    

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

    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\sklearn\ensemble\forest.py:246: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)
    

    准确率: 0.8071111111111111
                  precision    recall  f1-score   support
    
               0       0.83      0.95      0.88      6995
               1       0.63      0.32      0.43      2005
    
       micro avg       0.81      0.81      0.81      9000
       macro avg       0.73      0.63      0.66      9000
    weighted avg       0.79      0.81      0.78      9000
    
    

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

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
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

    准确率: 0.7517777777777778
    

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


```python

```


```python

```
