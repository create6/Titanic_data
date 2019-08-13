# coding: utf-8
import pandas as pd
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import calinski_harabaz_score,silhouette_score

# 加载数据
data = pd.read_csv('data.csv')
# print(data.head())
x_train = data[["2019年国际排名","2018世界杯","2015亚洲杯"]]

# 特征工程 - 标准化
min_max_scaler=preprocessing.StandardScaler()
x_train=min_max_scaler.fit_transform(x_train)

# 创建KMeans聚类评估器
estimator = KMeans(n_clusters=3)

# 模型训练
estimator.fit(x_train)
y_predict = estimator.predict(x_train)

# 模型评估
# CH系数
# 值越大聚类效果越好
print('CH系数:', calinski_harabaz_score(x_train, y_predict))
# 平均轮廓系数的取值范围为[-1,1]，系数越大，聚类效果越好
print('平均轮廓系数:', silhouette_score(x_train, y_predict))

# 合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(y_predict)),axis=1)
result.rename({0:u'聚类'},axis=1,inplace=True)
print(result)
