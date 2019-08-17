# 导入K近邻分类算法
from sklearn.neighbors import KNeighborsClassifier


# 1. 准备数据
x = [[0],[1],[2],[3]]
y = [0, 0, 1, 1]

# 2. 机器学习(模型训练)
# 创建评估器
estimator = KNeighborsClassifier(n_neighbors=3)
# 模型训练
estimator.fit(x, y)

# 3. 使用模型进行预测
# 预测: predict
rs = estimator.predict([[-1],[5], [2]])
print(rs)

