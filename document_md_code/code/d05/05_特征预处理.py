import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. 加载数据集
dating = pd.read_csv('data/dating.txt')
# print(dating)
# 无量化处理, 只需要处理特征, 无需处理目标值
# 获取特征值
x = dating[['milage', 'Liters', 'Consumtime']]
# print(x)
# 归一化
# # 2. 创建归一化转换器
# transfer = MinMaxScaler()
# # 3. 调用方法对数据进行归一化处理
# # x = transfer.fit_transform(x)
# # 扩展
# # 计算了每列数据的最大值和最小值
# transfer.fit(x)
#
# # 使用最大值和最小值进行归一化转换
# x = transfer.transform(x)
# print(x)
#

# 标准化: (x - 均值) / 标准差
# 把数据都放到均值为0, 方差为1的分布内
# 创建标准化转换器
transfer = StandardScaler()

# 计算每列数据的均值和标准差
# transfer.fit()
# 使用标准化对数据进行处理
# transfer.transform()

# 对数据进行标准化
x = transfer.fit_transform(x)
# 打印
print(x)



