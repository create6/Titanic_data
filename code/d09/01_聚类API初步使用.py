import matplotlib.pyplot as plt

from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import calinski_harabaz_score


# 1. 生成数据
X, y = make_blobs(n_samples=1000, n_features=2,
                  centers=[(-1,-1), (0, 0), (1, 1), (2,2)],
                  cluster_std=[0.4, 0.2, 0.2, 0.2]
                  )

print(X)

# 2. 使用KMeans进行聚类
estimator = KMeans(n_clusters=4)
y_pred = estimator.fit_predict(X)

# 3. 数据可视化(绘制分类结果图)
plt.figure(figsize=(5, 4), dpi=80)
# 绘制散点图, 查看数据的分布情况
plt.scatter(X[:,0], X[:,1], c=y_pred)
# 显示
plt.show()


# 分类评估
# 2 3096.7473856516135
# 3 2940.6149446783725
# 4 5866.614435267102
# CH系数: 值越大聚类效果越好.
print('CH系数', calinski_harabaz_score(X, y_pred))

