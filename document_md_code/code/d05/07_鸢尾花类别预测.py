from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# 1. 加载数据集
iris = load_iris()

# 2. 数据基本处理(分割数据集)
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=6)

# 3. 特征工程(特征预处理-标准化)
transfer = StandardScaler()
x_train = transfer.fit_transform(x_train)
# fit_transform: 使用x_test数据集的均值和标准差
# x_test = transfer.fit_transform(x_test)
# transform: 使用x_train数据集均值和标准差
x_test = transfer.transform(x_test)

# 4. 机器学习(模型训练): K近邻(KNN)
# 创建评估器对象
estimator = KNeighborsClassifier()

# 网格搜索和交叉验证进行参数调优
# 超参数字典
param_grid = {'n_neighbors': [3, 5, 7, 9]}

estimator = GridSearchCV(estimator, param_grid=param_grid, cv=5)


# 模型训练 训练集特征值, 训练集的目标值
estimator.fit(x_train, y_train)

# 5. 模型评估
# 使用模型对测试集进行预测
# 参数为测试集的特征值, 返回值为预测目标值
y_pre = estimator.predict(x_test)
print(y_pre)
print(y_pre == y_test)
# 准确率
score = estimator.score(x_test, y_test)
print('准确率:', score)

print('='*10)
# 网格搜索与交叉验证结果
print('交叉验证最好分数', estimator.best_score_)
print('交叉验证最好模型', estimator.best_estimator_)
print('交叉验证的结果 \n', estimator.cv_results_)







