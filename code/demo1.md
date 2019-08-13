
#### 机器学习练习
##### matplotlib


```python
import random

for i in range(3):
    a =random.randint(10,39)
    print(a)
    
```

    23
    20
    25


#### 画图


```python
import matplotlib.pyplot as plt
#1 创建画布
plt.figure(figsize=(20,8),dpi=100)
#2,画图
x =[1,2,3,4,5,6]
y=[12,11,6,23,12,9]
#绘制拆线
plt.plot(x,y)
#4展示
# plt.show()
```




    [<matplotlib.lines.Line2D at 0x1a5709eb4e0>]




![png](output_3_1.png)



```python
import matplotlib.pyplot as plt
import random

#准备数据,x,y参数
x=range(1,31)
y =[random.randint(12,39) for i in range(30)] #列表推导式 [random.randint(12,39) for i in x]
#创建画布
plt.figure(figsize=(17,8),dpi=100)
#绘制
#拆线图
# plt.plot(x,y)
#散点图
# plt.scatter(x,y)
#柱状图
plt.bar(x,y)

#添加描述信息
plt.xlabel('时间time')
plt.ylabel('temperature')
plt.title('time-temperature')
#save
# plt.savefig('plot{name}.png'.format('name',str(random.randint(100)))

#
plt.show()
```


![png](output_4_0.png)



```python

```


```python

```


```python

```


```python
import numpy as np
a = np.array([
    [
        [3.4,5,6,8],
        [3,2.4,5,7]
    ],
    [
        [2.3,4,5,6],
        [0.9,5,6,1]
    ],
    [
        [9,6.7,3,2],
        [1,3,4,5]
        ]
    ])
# 数组元素总个数
a.size

```




    24




```python
# 查看数组维度
a.ndim
```




    3




```python
 # 数组形状
a.shape
```




    (3, 2, 4)




```python
# 数组元素数据类型
a.dtype
```




    dtype('float64')




```python
ndarray01 = np.array([
    [
        [1,2,3,5],
        [3,4,5,6],
        [4,5,6,7],
        [9,4,5,6]
    ],
    [
        [9,8,4,5],
        [4,6,7,9],
        [9,5,3,1],
        [7,5,6,1]
    ]
])
#,dtype=float
```


```python
ndarray01[0]
```




    array([[1., 2., 3., 5.],
           [3., 4., 5., 6.],
           [4., 5., 6., 7.],
           [9., 4., 5., 6.]])




```python
ndarray01[1]
```




    array([[9., 8., 4., 5.],
           [4., 6., 7., 9.],
           [9., 5., 3., 1.],
           [7., 5., 6., 1.]])




```python
ndarray01.dtype
```




    dtype('float64')




```python
ndarray02 = np.array([
    [
        [1,2,3,5],
        [3,4,5,6],
        [4,5,6,7],
        [9,4,5,6]
    ],
    [
        [9, 8, 4, 5],
        [4, 6, 7, 9],
        [9, 5, 3, 1],
        [7, 5, 6, 1]
    ]
    ])
ndarray03= np.array(
    [
        [9,8,4,5],
        [4,6,7,9],
        [9,5,3,1],
        [7,5,6,1]
    ],dtype=float
)

```


```python
ndarray02[0]
```




    array([[1, 2, 3, 5],
           [3, 4, 5, 6],
           [4, 5, 6, 7],
           [9, 4, 5, 6]])




```python
ndarray03[0]
```




    array([9., 8., 4., 5.])




```python
# 创建指定长度或者形状的全0数组
np_zero = np.zeros((3,4))
np_zero
```




    array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])




```python
# 创建指定长度或形状的全1数组
np_ones = np.ones((4,6))
np_ones
```




    array([[1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1.]])




```python
# 数组变形，元素总数不变
np_ones_re = np_ones.reshape(2,12)
np_ones_re
```




    array([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
           [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])




```python
# 创建一个没有任何具体值的数组
np_empty = np.empty((2,3,4))
np_empty
```




    array([[[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]],
    
           [[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]]])




```python
np_empty2 = np.empty((2,3))
np_empty2
```




    array([[15., 20., 25.],
           [30., 35., 40.]])




```python
# # ndarray 的其他创建方式

# #arrange函数：类似于python的range函数，通过指定开始值、终值和步长来创建一维数组，注意数组不包括[终值]，
np_arange = np.arange(3,18,3)
np_arange
```




    array([ 3,  6,  9, 12, 15])




```python
np_arange = np.arange(3,18,5)
np_arange
```




    array([ 3,  8, 13])




```python

```
