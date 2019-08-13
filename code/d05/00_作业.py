# 作业第1题
import pandas as pd
import numpy as np

df=pd.DataFrame({"A":[5,3,None,4],
                 "B":[None,2,4,3],
                 "C":[4,3,8,5],
                 "D":[5,4,2,None]})
for i in df.columns:
    if np.all(pd.notnull(df[i])) == False:
        print(i)
        df[i].fillna(df[i].mean(), inplace=True)

print(df)

print('='*30)

# 作业第2题
# 2. 读取scores.txt文件中的数据, 获取所有同学语文成绩最低的1次考试成绩
# 1. 读取txt以空格或\t相隔的数据
scores = pd.read_table('./data/scores.txt', sep='\s+')
print(scores)
# 2. 定义函数获取按指定列升序排列的前几条数据
def top(x, n=1, column='chinese'):
    """
    x: 数据集
    n: 前几条数据, 默认 1
    column: 排序的列, 默认 chinese
    """
    return x.sort_values(by=column)[:n]

# 按名称分组, 获取按chinese升序排列的第一条数据
rs = scores.groupby('name').apply(top)
print(rs)

