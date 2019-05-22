import numpy as np
import pandas as pd
import time

print('-------np-------')
# print(help(np.sort))
# np
# a = np.array([
#     [
#         [3.4,5,6,8],
#         [3,2.4,5,7]
#     ],
#     [
#         [2.3,4,5,6],
#         [0.9,5,6,1]
#     ],
#     [
#         [9,6.7,3,2],
#         [1,3,4,5]
#         ]
#     ])
# # 查看数组维度
# print(a.ndim)
# # 数组元素数据类型
# print(a.dtype)
#
# # 数组形状
# print(a.shape)
# # 数组元素总个数
# print(a.size)
#
# ndarray01 = np.array([
#     [
#         [1,2,3,5],
#         [3,4,5,6],
#         [4,5,6,7],
#         [9,4,5,6]
#     ],
#     [
#         [9,8,4,5],
#         [4,6,7,9],
#         [9,5,3,1],
#         [7,5,6,1]
#     ]
# ],dtype=float)
# print(ndarray01[0])
# print(ndarray01[1])
#
# # ndarry的shape属性巧算
# # ndarray02 = np.array(
# #     [
# #         [1,2,3,5],
# #         [3,4,5,6],
# #         [4,5,6,7],
# #         [9,4,5,6]
# #     ],
# #     [
# #         [9, 8, 4, 5],
# #         [4, 6, 7, 9],
# #         [9, 5, 3, 1],
# #         [7, 5, 6, 1]
# #     ])
#
# # ndarray03= np.array(
# #     [
# #         [9,8,4,5],
# #         [4,6,7,9],
# #         [9,5,3,1],
# #         [7,5,6,1]
# #     ],dtype=float
# # )
# # print(ndarray02[0])
# # print(ndarray03[0])
# # 创建指定长度或者形状的全0数组
# np_zero = np.zeros((3,4))
# print(np_zero)
# # 创建指定长度或形状的全1数组
# np_ones = np.ones((4,6))
# print(np_ones)
# print('*'*30)
# # 创建一个没有任何具体值的数组
# np_empty = np.empty((2,3,4))
# print(np_empty)
# print('*'*30)
# # 数组变形，元素总数不变
# np_ones_re = np_ones.reshape(2,12)
# print(np_ones_re)
#
# # ndarray 的其他创建方式
#
# #arrange函数：类似于python的range函数，通过指定开始值、终值和步长来创建一维数组，注意数组不包括[终值]，
# np_arange = np.arange(3,18,3)
# print(np_arange)
# print('*'*30)
# #linspace函数：通过指定开始值、终值和元素个数来创建一维数组，可以通过endpoint关键字指定是否包括终值，缺省设置是包括终值
# np_lin = np.linspace(0,2,9)
# print(np_lin)
# #logspace函数：和linspace类似，不过它创建等比数列,使用随机数填充数组，即使用numpy.random模块的random()函数，数组所包含的的元素数量由参数决定'''
# # 10的幂函数
# np_log = np.logspace(0,2,3)
# print(np_log)
# #随机生成
# np_random = np.random.random((2,3,4)) #
# print(np_random)
# np_randint = np.random.randint(1,100,(2,3,4),'int64')  # 参数中dtype 缺损 ‘int’ :The default value is 'np.int'
# print(np_randint)
# #  查看类型
# print(np_randint.dtype)
# # 类型转换
# np_randint2 = np_randint.astype(float)
# print(np_randint2.dtype)

# NumPy中所支持的数据类型
# d = np.array(['Python','Scala','Java','C#'])
# print(d)
# print(d.dtype)
# e = np.array(['Python','Scala','Java','C++'],dtype=np.string_)
# print(e)
# print(e.dtype)
# e_2 = np.array(['Python','Scala','Java','C++'],dtype='S8')
# print(e_2)
# print(e_2.dtype)

# 改变ndarray 的形状
# 直接修改ndarray 的Shape值
# 使用reshape函数
# 当指定新数组某个轴的元素为-1时，将根据数组元素的个数自动计算此轴的长度

# a_arr = np.arange(0,20,2)
# print(a_arr)
# print(a_arr.size)
# a_arr = a_arr.reshape(2,5)
# print(a_arr)
# a_arr = a_arr.reshape(-1,5)
# print(a_arr)
# a_arr.shape = 5,-1
# print(a_arr)

# NumPy基本操作
#数组与标量之间的运算
# arr1 = np.array([1,2,3,4,5])
# arr1 = arr1 + 2
# print(arr1)
# arr1 = arr1 -3
# print(arr1)
# arr1 = arr1 *2
# print(arr1)
# arr1 = arr1/4
# print(arr1)
# arr1 = arr1 ** 2
# print(arr1)

# shape 相同的数组之间的运算
# ar_s1 = np.array([[1,2.0],[1.9,3.4]])
# ar_s2 = np.array([[3.6,1.2],[2.0,1.2]])
# ar_s3 = ar_s1 + ar_s2
# print(ar_s3)
# ar_s3 = ar_s1 - ar_s2
# print(ar_s3)
# ar_s3 = ar_s1 * ar_s2   # 对应位置参数相乘，不是矩阵乘法
# print(ar_s3)

# 元素级运算，在NumPy中，维度相同的数组之间运算，为元素级运算
#代码同上，相同shape(维度)

# 数组的矩阵积 np.dop
# ar_s1 = np.array([[1,2.0],[1.9,3.4]])
# ar_s2 = np.array([[3.6,1.2],[2.0,1.2]])
# ar_sj = ar_s1.dot(ar_s2)
# ar_sj2 = np.dot(ar_s1,ar_s2)
# ar_sj3 = np.dot(ar_s2,ar_s1)
# print(ar_sj)
# print(ar_sj2)
# print(ar_sj3)

# 数组的索引与切片










print('-------pd-------')
# 【1】导入数据库或者创建数据表
# df = pd.DataFrame(pd.read_csv('train_2.csv',header =1))
# df = pd.DataFrame(pd.read_excel('name.xlsx'))

df = pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006],"date":pd.date_range('20130102', periods=6), "city":['Beijing ',
       'S H       ', ' guangzhou ', '   Shenzhen            ', '                  shanghai', 'Beijing '], "age":[23,44,54,32,34,32], "category":['100-A','100-B',
      '110-A','110-C','210-A','130-F'], "price":[1200,np.nan,2133,5433,np.nan,4432]}, columns =['id',
         'date','city','category','age','price'])

#-----[2]查看

#1 查看数据表的维度
print(df.shape)

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


# ---- 【3】数据清洗，都需要重新赋值

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
# df_replace = df['city'].replace('S H','ShangHai') # 替换后用info()会报错！
# print(df_replace.head(6))
# df_replace.info()

# ------【4】数据预处理
# 1.数据表合并

df1 = pd.DataFrame({"id":[1001,1002,1003,1004,1005,1006,1007,1008],
                  "gender":['male','female','male','female','male','female','male','female'],
                  "pay":['Y','N','Y','Y','N','Y','N','Y',], "m-point":[10,12,20,40,40,40,30,20]})

print(df1.shape)

# df1.info()
# 数据表匹配合并
df_inner = pd.merge(df,df1,how='inner')
# df_inn1 = pd.merge(df,df1,how='left')
# df_inn2 = pd.merge(df,df1,how='right')
# df_inn3 = pd.merge(df,df1,how='outer')
#print(df_inner)
# print(df_inn1)
# print(df_inn2)
# print(df_inn3)

# 计时装饰器，带参数
def show_time(str1):
    def show_time_(func):
        print('-'* 30)
        print( '%s开始计时'%str1)
        def fn_in(*args,**kwargs):
            t1 = time.time()
            func(*args,**kwargs)
            t2 = time.time()
            print('所用时间:%f'%(t2 - t1))
        return fn_in
    return show_time_
# 测试 计时器
# @show_time('s_test')  # s_test()= show_time(s_test)
# def s_test():
#     print(df_inner.head(8))
# s_test()

#2. 设置索引列
df_inner_index = df_inner.set_index('id')
# @show_time('set_index')  # set_index()= show_time(set_index)
# def set_index():
#     df_inner1 = df_inner.set_index('id') # 在函数内不能用同名进行重新赋值，即不能再用df_inner
#     print(df_inner1.head(8))
# set_index()


# 按特定列的值排序
# @show_time('inner_sort')
# def inner_sort():
#     df_inner_sort = df_inner.sort_values(by=['age']) # 在函数内不能用同名进行重新赋值
#     print(df_inner_sort)
# inner_sort()

# 按索引列排序 # 比非索引列排序要快
# @show_time('sort_index')
# def sort_index():
#     df_inner_sd = df_inner_index.sort_index()
#
#     print(df_inner_sd)
# sort_index()

# 数据分组
# 如果price 列的值>3000 ，group 列显示为high ，否则显示low
#df_inner['group'] = np.where(df_inner['price']>3000,'high','low')
# print(df_inner)
# print(df_inner.head(6))

# 对复合多个条件的数据进行分组标记
#df_inner.loc[(df_inner['city'] == 'Beijing')&(df_inner['price']>4000),'sign']=1
# print(df_inner)
