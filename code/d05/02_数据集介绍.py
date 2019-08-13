from sklearn.datasets import load_iris, fetch_20newsgroups

# 加载小数据集
# 直接从本地加载
iris = load_iris()
# print(iris)
# print(dir(iris))
# print(type(iris)) # sklearn.utils.Bunch
# # 常用属性
# print('特征值数据:\n', iris.data)
# print('目标值数据:\n', iris.target)
# print('特征值名称:\n', iris.feature_names)
# print('目标值名称:\n', iris.target_names)
# print('数据集描述:\n', iris.DESCR)

# 加载大数据集: (需要从网上下载; 下载一次)
news = fetch_20newsgroups(subset='all')
print(news)









