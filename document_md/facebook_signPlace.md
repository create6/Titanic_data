

```python
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
```


```python
facebook = pd.read_csv("D:/003_IT/FBlocation/train.csv")
print(facebook.head())

```

       row_id       x       y  accuracy    time    place_id
    0       0  0.7941  9.0809        54  470702  8523065625
    1       1  5.9567  4.7968        13  186555  1757726713
    2       2  8.3078  7.0407        74  322648  1137537235
    3       3  7.3665  2.5165        65  704587  6567393236
    4       4  4.0961  1.1307        31  472130  7440663949
    


```python
facebook_data=facebook.query('x>2.0 & x<2.5 & y>2.0 & y <2.5')
#时间转换
date_time = pd.to_datetime(facebook_data['time'],unit='s')
date_time =pd.DatetimeIndex(date_time)
#添加时间特征的列
facebook_data['day']=date_time.day
facebook_data['weekday']=date_time.weekday
facebook_data['hour']=date_time.hour

#去掉签到位置少的地方

#统计每一个位置签到数量
place_counts = facebook_data.groupby('place_id').count()
#选择签到位置大于3次的地方
place_counts=place_counts[place_counts['row_id']>3]

#从数据中选择签到位置大于3的位置
facebook_data =facebook_data[facebook_data['place_id'].isin(place_counts.index)]

#确定特征值和目标值
x =facebook_data[['x','y','accuracy','day','weekday','hour']]
y =facebook_data['place_id']
#分割数据集
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=8) 



```

    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\ipykernel_launcher.py:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\ipykernel_launcher.py:7: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      import sys
    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\ipykernel_launcher.py:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      
    


```python
#特征工程(标准化)
transfer = StandardScaler()
x_train =transfer.fit_transform(x_train)
x_test = transfer.fit_transform(x_test)

#机器学习 KNN+GridSearchCV
estimator =KNeighborsClassifier()


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

    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\sklearn\preprocessing\data.py:645: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\sklearn\base.py:464: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)
    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\sklearn\preprocessing\data.py:645: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.partial_fit(X, y)
    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\sklearn\base.py:464: DataConversionWarning: Data with input dtype int64, float64 were all converted to float64 by StandardScaler.
      return self.fit(X, **fit_params).transform(X)
    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\sklearn\model_selection\_split.py:652: Warning: The least populated class in y has only 1 members, which is too few. The minimum number of members in any class cannot be less than n_splits=5.
      % (min_groups, self.n_splits)), Warning)
    

    准确率: 0.35832531280076996
    交叉验证最好分数: 0.3537868162692847
    交叉验证最好的模型: KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
               metric_params=None, n_jobs=None, n_neighbors=5, p=2,
               weights='uniform')
    交叉验证的结果:
     {'mean_fit_time': array([0.0971561 , 0.06995711, 0.07115345, 0.07135506]), 'std_fit_time': array([0.02997233, 0.00909344, 0.01557926, 0.0107372 ]), 'mean_score_time': array([0.84228096, 0.84707961, 0.86246977, 0.99029398]), 'std_score_time': array([0.05086247, 0.12299685, 0.03593845, 0.26202955]), 'param_n_neighbors': masked_array(data=[3, 5, 7, 9],
                 mask=[False, False, False, False],
           fill_value='?',
                dtype=object), 'params': [{'n_neighbors': 3}, {'n_neighbors': 5}, {'n_neighbors': 7}, {'n_neighbors': 9}], 'split0_test_score': array([0.33038289, 0.34281452, 0.33843859, 0.33814023]), 'split1_test_score': array([0.33848954, 0.34880194, 0.34607219, 0.34121929]), 'split2_test_score': array([0.33888717, 0.35236038, 0.35277178, 0.34464671]), 'split3_test_score': array([0.34521556, 0.3617245 , 0.36014721, 0.35047319]), 'split4_test_score': array([0.35196131, 0.36432026, 0.36120365, 0.35314347]), 'mean_test_score': array([0.34079284, 0.35378682, 0.3514974 , 0.34537167]), 'std_test_score': array([0.00721528, 0.00800611, 0.00862862, 0.00558255]), 'rank_test_score': array([4, 1, 2, 3]), 'split0_train_score': array([0.6073278 , 0.54130995, 0.50605012, 0.47461552]), 'split1_train_score': array([0.60575752, 0.53820641, 0.49962428, 0.47257275]), 'split2_train_score': array([0.60805965, 0.53896958, 0.50179304, 0.47186605]), 'split3_train_score': array([0.60368451, 0.53692205, 0.50115462, 0.47128855]), 'split4_train_score': array([0.60348656, 0.53602695, 0.4980219 , 0.46846525]), 'mean_train_score': array([0.60566321, 0.53828699, 0.50132879, 0.47176162]), 'std_train_score': array([0.00185341, 0.00182107, 0.00269732, 0.00199517])}
    


```python

```


```python

```


```python

```
