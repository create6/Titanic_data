from sklearn.linear_model import LinearRegression

# 准备数据
x = [[80, 86],
[82, 80],
[85, 78],
[90, 90],
[86, 82],
[82, 90],
[78, 80],
[92, 94]]
y = [84.2, 80.6, 80.1, 90, 83.2, 87.6, 79.4, 93.4]
# 机器学习(模型训练)
estimator = LinearRegression()
estimator.fit(x, y)

print('回归系数', estimator.coef_)

# 预测
y_pre = estimator.predict([[100, 80], [80, 86]])
print('预测值', y_pre)


