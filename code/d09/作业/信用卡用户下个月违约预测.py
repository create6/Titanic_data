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

print(x_train)

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
    print(result)


