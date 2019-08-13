from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor, Ridge
from sklearn.externals import joblib


# mean_squared_error(均方误差): (预测值-真实值)平方求和/样本数量
# mean_absolute_error(平均绝对误差)(预测值-真实值)绝对值求和/样本数量
from sklearn.metrics import mean_squared_error, mean_absolute_error


def linear_regression():
    """正规方程"""
    # 1. 加载数据集
    data = load_boston()
    # 2. 数据的基本处理(分割数据集)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.data, data.target, test_size=0.25, random_state=8)

    # 3. 特征工程(特征预处理, 标准化)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test =  transfer.transform(x_test)

    # 4. 机器学习(模型训练): LinearRegression
    estimator = LinearRegression()
    estimator.fit(x_train, y_train)

    # 5. 模型评估
    print('模型系数', estimator.coef_)
    print('模型偏置', estimator.intercept_)
    # 预测
    y_pre = estimator.predict(x_test)
    print('预测值 - 真实值', y_pre - y_test)

    # 模型评估
    # 均方误差
    mse = mean_squared_error(y_test, y_pre)
    print('均方误差', mse)
    # 平均绝对误差
    mae = mean_absolute_error(y_test, y_pre)
    print('平均绝对误差', mae)


def linear_SGDRegressor():
    """正规方程"""
    # 1. 加载数据集
    data = load_boston()
    # 2. 数据的基本处理(分割数据集)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.data, data.target, test_size=0.25, random_state=8)

    # 3. 特征工程(特征预处理, 标准化)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test =  transfer.transform(x_test)

    # 4. 机器学习(模型训练): SGDRegressor
    estimator = SGDRegressor()
    # estimator = SGDRegressor(learning_rate='constant', eta0=0.01)

    estimator.fit(x_train, y_train)

    # 5. 模型评估
    print('模型系数', estimator.coef_)
    print('模型偏置', estimator.intercept_)
    # 预测
    y_pre = estimator.predict(x_test)
    print('预测值 - 真实值', y_pre - y_test)

    # 模型评估
    # 均方误差
    mse = mean_squared_error(y_test, y_pre)
    print('均方误差', mse)
    # 平均绝对误差
    mae = mean_absolute_error(y_test, y_pre)
    print('平均绝对误差', mae)


def linear_Ridge():
    """岭回归"""
    # 1. 加载数据集
    data = load_boston()
    # 2. 数据的基本处理(分割数据集)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.data, data.target, test_size=0.25, random_state=8)

    # 3. 特征工程(特征预处理, 标准化)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test =  transfer.transform(x_test)

    # 4. 机器学习(模型训练):  Ridge
    estimator = Ridge(alpha=1.0)

    estimator.fit(x_train, y_train)

    # 5. 模型评估
    print('模型系数', estimator.coef_)
    print('模型偏置', estimator.intercept_)
    # 预测
    y_pre = estimator.predict(x_test)
    print('预测值 - 真实值', y_pre - y_test)

    # 模型评估
    # 均方误差
    mse = mean_squared_error(y_test, y_pre)
    print('均方误差', mse)
    # 平均绝对误差
    mae = mean_absolute_error(y_test, y_pre)
    print('平均绝对误差', mae)



def model_save_load():
    """模型保存于加载"""
    # 1. 加载数据集
    data = load_boston()
    # 2. 数据的基本处理(分割数据集)
    x_train, x_test, y_train, y_test = \
        train_test_split(data.data, data.target, test_size=0.25, random_state=8)

    # 3. 特征工程(特征预处理, 标准化)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test =  transfer.transform(x_test)

    # 4. 机器学习(模型训练):  Ridge
    # estimator = Ridge(alpha=1.0)
    #
    # estimator.fit(x_train, y_train)

    # 保存模型
    # joblib.dump(estimator, 'test.pkl')
    estimator = joblib.load('test.pkl')

    # 5. 模型评估
    print('模型系数', estimator.coef_)
    print('模型偏置', estimator.intercept_)
    # 预测
    y_pre = estimator.predict(x_test)
    print('预测值 - 真实值', y_pre - y_test)

    # 模型评估
    # 均方误差
    mse = mean_squared_error(y_test, y_pre)
    print('均方误差', mse)
    # 平均绝对误差
    mae = mean_absolute_error(y_test, y_pre)
    print('平均绝对误差', mae)


if __name__ == '__main__':
    # linear_regression()
    # linear_SGDRegressor()
    # linear_Ridge()
    model_save_load()