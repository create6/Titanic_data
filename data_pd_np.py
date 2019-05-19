import numpy as np
import pandas as pd

# 导入数据库或者创建数据表
# df = pd.DataFrame(pd.read_csv('train_2.csv',header =1))
# df = pd.DataFrame(pd.read_excel('name.xlsx'))

df = pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006],"date":pd.date_range('20130102', periods=6), "city":['Beijing ',
       'S H       ', ' guangzhou ', '   Shenzhen            ', '                  shanghai', 'Beijing '], "age":[23,44,54,32,34,32], "category":['100-A','100-B',
      '110-A','110-C','210-A','130-F'], "price":[1200,np.nan,2133,5433,np.nan,4432]}, columns =['id',
         'date','city','category','age','price'])

#1 查看数据表的维度
# print(df.shape)

#2 查看数据表信息
# df.info()

#3 查看数据表概况
#print(df.describe())

#4 查看数据表各列格式
# print(df.dtypes)

#5 查看单列格式
# print(df['date'].dtype)

# 6检查数据表空值
# print(df.isnull())

# 7 检查特定列空值
# print(df['price'].isnull())

# 8 查看city列中的唯一值
# print(df['city'].unique())

# 9 查看数据表的值
# print(df.values)

# 10 查看列名称
# print(df.columns)

# 11 显示前N行，默认前10行
# print(df.head())

# 12 查看前3行数据
# print(df.head(3))

# 13 查看最后3行
# print(df.tail(3))

# ---- 数据清洗

# 删除数据表中含有空值的行

# df = df.dropna(how = 'any')  # 改动需要重新赋值
# print(df.values)

# 使用数字0填充数据中空值
# df = df.fillna(value=0)  # 填充后需要重新赋值
# print(df.values)

# 使用prince的均值对NA进行填充
# print(df['price'].mean())
df['price'] = df['price'].fillna(df['price'].mean())  # 填充后需要重新赋值
# print(df.head(6))

# 清除city字段的字符空格
# print(df.head(6))
df['city'] = df['city'].map(str.strip)    # 字符内部的空格不会被去掉
# print(df.head(6))

# city列大小写转换 lower()  upper()
# print(df.head(6))
# df['city'] = df['city'].str.lower()
# df['city'] = df['city'].str.upper()
# print(df.head(6))

# 更改数据格式
df['price'] = df['price'].astype('int')
# print(df.head(6))

# 更改列名称
df = df.rename(columns ={'category':'category_size'})  # 赋值
# print(df.head(6))

# print(df['city'])
# 删除后出现的重复值
# df = df['city'].drop_duplicates()
# print(df.head(6))

# 删除先出现的重复值
# df = df['city'].drop_duplicates(keep= 'last')
# print(df.head(6))

# 数据替换
df = df['city'].replace('S H','ShangHai')
print(df.head(6))

# 数据预处理

# 1.数据表合并

#2. 设置索引列
