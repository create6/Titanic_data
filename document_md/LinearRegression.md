
### 简单线性回归


```python
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = [[80, 86],
[82, 80],
[85, 78],
[90, 90],
[86, 82],
[82, 90],
[78, 80],
[92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]

# plt.figure(figsize=(20,8),dpi=8)
# plt.scatter(x)
# plt.show()

#实例化API
estimator = LinearRegression()
#使用fit方法进行训练
estimator.fit(x,y)
print('回归系数:',estimator.coef_)
y_pre =estimator.predict([[100,80]])
print('预测值:',y_pre)
```

    回归系数: [0.3 0.7]
    预测值: [86.]
    

### 案例：波士顿房价预测


```python
from sklearn.datasets import load_boston  #数据
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,SGDRegressor #线性回归模型与梯度下降模型(随机梯度下降算法)
from sklearn.metrics import mean_squared_error,mean_absolute_error    
#mean_squared_error均方误差,mean_absolute_error平均均方误差


def pre_data():
    #加载数据
    data = load_boston()
    #数据分割
    x_train,x_test,y_train,y_test =train_test_split(data.data,data.target,test_size=0.25,random_state=8)
    #特征工程,标准化
    transfer =StandardScaler()
    x_train=transfer.fit_transform(x_train)
    x_test=transfer.fit_transform(x_test)
    
    return x_train,x_test,y_train,y_test


def linear_regression():
    ''' 线性回归:正规方程'''

    x_train,x_test,y_train,y_test =pre_data()
    #机器学习 
    estimator = LinearRegression()
    estimator.fit(x_train,y_train)
    
    #模型评估
    print('模型系数:',estimator.coef_)
    print('模型偏置:',estimator.intercept_)
    #预测
    y_pre = estimator.predict(x_test)
#     print('预测值-真实值:',y_pre-y_test)

    #模型评估
    mse =mean_squared_error(y_test,y_pre)
    print('均方误差:',mse)
    #平均绝对误差
    mae = mean_absolute_error(y_test,y_pre)
    print('平均绝对误差:',mae)
    print('-'*20)
    
def c_SGDRegressorDR():
    ''' 线性回归:梯度下降'''

    x_train,x_test,y_train,y_test =pre_data()
    #机器学习
#     estimator = LinearRegression()
    estimator = SGDRegressor()
    estimator.fit(x_train,y_train)
    
    #模型评估
    print('模型系数:',estimator.coef_)
    print('模型偏置:',estimator.intercept_)
    #预测
    y_pre = estimator.predict(x_test)
#     print('预测值-真实值:',y_pre-y_test)

    #模型评估
    mse =mean_squared_error(y_test,y_pre)
    print('均方误差:',mse)
    #平均绝对误差
    mae = mean_absolute_error(y_test,y_pre)
    print('平均绝对误差:',mae)
    
    
linear_regression()
c_SGDRegressorDR()
    
    

```

    模型系数: [-0.98162265  1.16064607  0.18611408  0.64865713 -1.48273565  2.67325335
     -0.16756838 -3.00571558  2.29915542 -1.83639913 -1.92095414  0.85800075
     -4.05354071]
    模型偏置: 22.52163588390508
    均方误差: 22.231973959150817
    平均绝对误差: 3.206908326834049
    --------------------
    模型系数: [-0.70572786  0.70901263 -0.29653019  0.81359123 -0.5511823   3.09446727
     -0.16463064 -2.06708888  0.89098569 -0.3940859  -1.72157882  0.80075617
     -3.61577583]
    模型偏置: [22.07511813]
    均方误差: 23.634341122128546
    平均绝对误差: 3.2320617646691923
    

    c:\users\struggle6\appdata\local\programs\python\python37\lib\site-packages\sklearn\linear_model\stochastic_gradient.py:166: FutureWarning: max_iter and tol parameters have been added in SGDRegressor in 0.19. If both are left unset, they default to max_iter=5 and tol=None. If tol is not None, max_iter defaults to max_iter=1000. From 0.21, default max_iter will be 1000, and default tol will be 1e-3.
      FutureWarning)
    


```python

```
