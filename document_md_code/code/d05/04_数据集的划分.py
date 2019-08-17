from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split



# 训练集: 用于模型训练的(特征值, 目标值)
# 测试集: 用于评估模型(特征值, 目标值)

# 加载数据
iris = load_iris()

# 分割数据集
# 训练集的特征值, 测试集的特征值, 训练集的目标值, 测试集的目标值
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=22)
# print('训练集的特征值', x_train)
# print('测试集的特征值', x_test)
# print('训练集的目标值', y_train)
print('测试集的目标值', y_test)

# print('训练集的特征值', x_train.shape)
# print('测试集的特征值', x_test.shape)
# print('训练集的目标值', y_train.shape)
# print('测试集的目标值', y_test.shape)

# random_state随机数种子, 如果种子相同分割数据集就相同, 如果不同分割数据集也不同
x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, test_size=0.3, random_state=22)
print('测试集的目标值', y_test1)
x_train1, x_test1, y_train1, y_test1 = train_test_split(iris.data, iris.target, test_size=0.3, random_state=6)
print('测试集的目标值', y_test1)

