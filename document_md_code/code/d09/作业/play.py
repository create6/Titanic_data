import pandas as pd
import numpy as np
# 否去打网球（play）主要由天气（outlook）、温度（temperature）、湿度（humidity）、是否有风（windy）来确定。样本中共14条数据。数据如下所示:
datas = pd.read_csv('play.txt',   sep='\s+', names=['id', 'outlook', 'temperature','humidity','windy','play' ])

# print(datas)
print(datas['play'] == 'no')
print(datas['play'].values.dtype)
new_data = np.where(datas['play'] == 'yes', 1, 0)


print(new_data)