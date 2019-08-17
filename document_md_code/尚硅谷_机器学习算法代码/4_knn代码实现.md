
# k近邻算法教程

### 0.引入依赖


```python
import numpy as np
import pandas as pd

# 这里直接引入sklearn里的数据集，iris鸢尾花
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split  # 切分数据集为训练集和测试集
from sklearn.metrics import accuracy_score # 计算分类预测的准确率
```

### 1. 数据加载和预处理


```python
iris = load_iris()

df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df['class'] = iris.target
df['class'] = df['class'].map({0: iris.target_names[0], 1: iris.target_names[1], 2: iris.target_names[2]})
df.head(10)
df.describe()
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
      <th>sepal length (cm)</th>
      <th>sepal width (cm)</th>
      <th>petal length (cm)</th>
      <th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.843333</td>
      <td>3.057333</td>
      <td>3.758000</td>
      <td>1.199333</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.828066</td>
      <td>0.435866</td>
      <td>1.765298</td>
      <td>0.762238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
x = iris.data
y = iris.target.reshape(-1,1)
print(x.shape, y.shape)
```

    (150, 4) (150, 1)
    


```python
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=35, stratify=y)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


a = np.array([[3,2,4,2],
             [2,1,4,23],
             [12,3,2,3],
             [2,3,15,23],
             [1,3,2,3],
             [13,3,2,2],
             [213,16,3,63],
             [23,62,23,23],
             [23,16,23,43]])
b = np.array([[1,1,1,1]])
print(a-b)
np.sum(np.abs(a - b), axis=1)
dist = np.sqrt( np.sum((a-b) ** 2, axis=1) )
print(dist)

```

    (105, 4) (105, 1)
    (45, 4) (45, 1)
    [[  2   1   3   1]
     [  1   0   3  22]
     [ 11   2   1   2]
     [  1   2  14  22]
     [  0   2   1   2]
     [ 12   2   1   1]
     [212  15   2  62]
     [ 22  61  22  22]
     [ 22  15  22  42]]
    [  3.87298335  22.22611077  11.40175425  26.17250466   3.
      12.24744871 221.39783197  71.92357055  54.3783045 ]
    

### 2. 核心算法实现


```python
# 距离函数定义
def l1_distance(a, b):
    return np.sum(np.abs(a-b), axis=1)
def l2_distance(a, b):
    return np.sqrt( np.sum((a-b) ** 2, axis=1) )

# 分类器实现
class kNN(object):
    # 定义一个初始化方法，__init__ 是类的构造方法
    def __init__(self, n_neighbors = 1, dist_func = l1_distance):
        self.n_neighbors = n_neighbors
        self.dist_func = dist_func
    
    # 训练模型方法
    def fit(self, x, y):
        self.x_train = x
        self.y_train = y
    
    # 模型预测方法
    def predict(self, x):
        # 初始化预测分类数组
        y_pred = np.zeros( (x.shape[0], 1), dtype=self.y_train.dtype )
        
        # 遍历输入的x数据点，取出每一个数据点的序号i和数据x_test
        for i, x_test in enumerate(x):
            # x_test跟所有训练数据计算距离
            distances = self.dist_func(self.x_train, x_test)
            
            # 得到的距离按照由近到远排序，取出索引值
            nn_index = np.argsort(distances)
            
            # 选取最近的k个点，保存它们对应的分类类别
            nn_y = self.y_train[ nn_index[:self.n_neighbors] ].ravel()
            
            # 统计类别中出现频率最高的那个，赋给y_pred[i]
            y_pred[i] = np.argmax( np.bincount(nn_y) )
        
        return y_pred


nn_index = np.argsort(dist)
print("dist: ",dist)
print("nn_index: ",nn_index)
nn_y = y_train[ nn_index[:9] ].ravel()
#print(y_train[:8])
print("nn_y: ",nn_y)
print(np.bincount(nn_y))
print(np.argmax(np.bincount(nn_y)))

```

    dist:  [  3.87298335  22.22611077  11.40175425  26.17250466   3.
      12.24744871 221.39783197  71.92357055  54.3783045 ]
    nn_index:  [4 0 2 5 1 3 8 7 6]
    nn_y:  [1 1 2 1 2 2 0 0 1]
    [2 4 3]
    1
    

### 3. 测试


```python
# 定义一个knn实例
knn = kNN(n_neighbors = 3)
# 训练模型
knn.fit(x_train, y_train)
# 传入测试数据，做预测
y_pred = knn.predict(x_test)

print(y_test.ravel())
print(y_pred.ravel())

# 求出预测准确率
accuracy = accuracy_score(y_test, y_pred)

print("预测准确率: ", accuracy)
```

    [2 1 2 2 0 0 2 0 1 1 2 0 1 1 1 2 2 0 1 2 1 0 0 0 1 2 0 2 0 0 2 1 0 2 1 0 2
     1 2 2 1 1 1 0 0]
    [2 1 2 2 0 0 2 0 1 1 1 0 1 1 1 2 2 0 1 2 1 0 0 0 1 2 0 2 0 0 2 1 0 2 1 0 2
     1 2 1 1 2 1 0 0]
    预测准确率:  0.9333333333333333
    


```python
# 定义一个knn实例
knn = kNN()
# 训练模型
knn.fit(x_train, y_train)

# 保存结果list
result_list = []

# 针对不同的参数选取，做预测
for p in [1, 2]:
    knn.dist_func = l1_distance if p == 1 else l2_distance
    
    # 考虑不同的k取值，步长为2
    for k in range(1, 10, 2):
        knn.n_neighbors = k
        # 传入测试数据，做预测
        y_pred = knn.predict(x_test)
        # 求出预测准确率
        accuracy = accuracy_score(y_test, y_pred)
        result_list.append([k, 'l1_distance' if p == 1 else 'l2_distance', accuracy])
df = pd.DataFrame(result_list, columns=['k', '距离函数', '预测准确率'])
df
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
      <th>k</th>
      <th>距离函数</th>
      <th>预测准确率</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>l1_distance</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>l1_distance</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>l1_distance</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7</td>
      <td>l1_distance</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9</td>
      <td>l1_distance</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>l2_distance</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>3</td>
      <td>l2_distance</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>7</th>
      <td>5</td>
      <td>l2_distance</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7</td>
      <td>l2_distance</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>l2_distance</td>
      <td>0.977778</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
