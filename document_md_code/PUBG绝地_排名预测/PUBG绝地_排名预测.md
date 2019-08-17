
# 《绝地求生》玩家排名预测
                   ---- 你能预测《绝地求生》玩家战斗结束后的排名吗？


![img1](./img/img1.png)

# 项目背景
## 项目简介
绝地求生(Player unknown's Battlegrounds)，俗称吃鸡，是一款战术竞技型射击类沙盒游戏。
这款游戏是一款大逃杀类型的游戏，每一局游戏将有最多100名玩家参与，他们将被投放在绝地岛(battlegrounds)上，在游戏的开始时所有人都一无所有。玩家需要在岛上收集各种资源，在不断缩小的安全区域内对抗其他玩家，让自己生存到最后。

该游戏拥有很高的自由度，玩家可以体验飞机跳伞、开越野车、丛林射击、抢夺战利品等玩法，小心四周埋伏的敌人，尽可能成为最后1个存活的人。


![img2](./img/img2.png)

## 项目涉及知识点
 - sklearn基本操作
 - 数据基本处理
 - 机器学习基本算法的使用


## 数据集介绍
本项目中，将为您提供大量匿名的《绝地求生》游戏统计数据。
其格式为每行包含一个玩家的游戏后统计数据，列为数据的特征值。
数据来自所有类型的比赛：单排，双排，四排；不保证每场比赛有100名人员，每组最多4名成员。

文件说明:

- train_V2.csv - 训练集

- test_V2.csv - 测试集


数据集局部图如下图所示:

![img3](./img/img3.png)

数据集中字段解释：

- Id [用户id]
  - Player’s Id
- groupId [所处小队id]
  - ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
- matchId [该场比赛id]
  - ID to identify match. There are no matches that are in both the training and testing set.
- assists [助攻数]
  - Number of enemy players this player damaged that were killed by teammates.
- boosts [使用能量,道具数量]
  - Number of boost items used.
- damageDealt [总伤害]
  - Total damage dealt. Note: Self inflicted damage is subtracted.
- DBNOs [击倒敌人数量]
  - Number of enemy players knocked.
- headshotKills [爆头数]
  - Number of enemy players killed with headshots.
- heals [使用治疗药品数量]
  - Number of healing items used.
- killPlace [本厂比赛杀敌排行]
  - Ranking in match of number of enemy players killed.
- killPoints [Elo杀敌排名]
  - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.) If there is a value other than -1 in rankPoints, then any 0 in killPoints should be treated as a “None”.
- kills [杀敌数]
  - Number of enemy players killed.
- killStreaks [连续杀敌数]
  - Max number of enemy players killed in a short amount of time.
- longestKill [最远杀敌距离]
  - Longest distance between player and player killed at time of death. This may be misleading, as downing a player and driving away may lead to a large longestKill stat.
- matchDuration [比赛时长] 
  - Duration of match in seconds.
- matchType [比赛类型(小组人数)]
  - String identifying the game mode that the data comes from. The standard modes are “solo”, “duo”, “squad”, “solo-fpp”, “duo-fpp”, and “squad-fpp”; other modes are from events or custom matches.
- maxPlace [本局最差名次]
  - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
- numGroups [小组数量]
  - Number of groups we have data for in the match.
- rankPoints [Elo排名]
  - Elo-like ranking of player. This ranking is inconsistent and is being deprecated in the API’s next version, so use with caution. Value of -1 takes place of “None”.
- revives [救活队员的次数]
  - Number of times this player revived teammates.
- rideDistance [驾车距离]
  - Total distance traveled in vehicles measured in meters.
- roadKills [驾车杀敌数]
  - Number of kills while in a vehicle.
- swimDistance [游泳距离]
  - Total distance traveled by swimming measured in meters.
- teamKills [杀死队友的次数]
  - Number of times this player killed a teammate.
- vehicleDestroys [毁坏机动车的数量]
  - Number of vehicles destroyed.
- walkDistance [步行距离]
  - Total distance traveled on foot measured in meters.
- weaponsAcquired [收集武器的数量]
  - Number of weapons picked up.
- winPoints [胜率Elo排名]
  - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.) If there is a value other than -1 in rankPoints, then any 0 in winPoints should be treated as a “None”.
- winPlacePerc [百分比排名]
  - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# 项目评估方式
## 评估方式
你必须创建一个模型，根据他们的最终统计数据预测玩家的排名，从1（第一名）到0（最后一名）。

最后结果通过平均绝对误差（MAE）进行评估，即通过预测的winPlacePerc和真实的winPlacePerc之间的平均绝对误差

## MAE(Maean Absolute Error)介绍
 - 就是绝对误差的平均值

 - 能更好地反映预测值误差的实际情况
     $$
     MAE(X,h) = \frac{1}{m} \sum_{i=1}^{m} {|h(x^{(i)}) - y^{(i)}|}
     $$


api:
 - sklearn.metrics.mean_absolute_error

# 项目实现（数据分析+RL）
在接下来的分析中，我们将分析数据集，检测异常值。

然后我们通过随机森林模型对其训练，并对对该模型进行了优化。


```python
# 导入数据基本处理阶段需要用到的api
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
```

## 获取数据、基本数据信息查看
导入数据，且查看数据的基本信息


```python
train = pd.read_csv("../data/train_V2.csv")
```


```python
new_train = train.dropna()
```


```python
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>killPoints</th>
      <th>kills</th>
      <th>killStreaks</th>
      <th>...</th>
      <th>revives</th>
      <th>rideDistance</th>
      <th>roadKills</th>
      <th>swimDistance</th>
      <th>teamKills</th>
      <th>vehicleDestroys</th>
      <th>walkDistance</th>
      <th>weaponsAcquired</th>
      <th>winPoints</th>
      <th>winPlacePerc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>...</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
      <td>4.446965e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2.338150e-01</td>
      <td>1.106908e+00</td>
      <td>1.307172e+02</td>
      <td>6.578757e-01</td>
      <td>2.268196e-01</td>
      <td>1.370148e+00</td>
      <td>4.759936e+01</td>
      <td>5.050062e+02</td>
      <td>9.247835e-01</td>
      <td>5.439553e-01</td>
      <td>...</td>
      <td>1.646590e-01</td>
      <td>6.061158e+02</td>
      <td>3.496092e-03</td>
      <td>4.509323e+00</td>
      <td>2.386841e-02</td>
      <td>7.918209e-03</td>
      <td>1.154218e+03</td>
      <td>3.660488e+00</td>
      <td>6.064603e+02</td>
      <td>4.728216e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5.885731e-01</td>
      <td>1.715794e+00</td>
      <td>1.707806e+02</td>
      <td>1.145743e+00</td>
      <td>6.021553e-01</td>
      <td>2.679982e+00</td>
      <td>2.746293e+01</td>
      <td>6.275049e+02</td>
      <td>1.558445e+00</td>
      <td>7.109721e-01</td>
      <td>...</td>
      <td>4.721671e-01</td>
      <td>1.498344e+03</td>
      <td>7.337297e-02</td>
      <td>3.050220e+01</td>
      <td>1.673935e-01</td>
      <td>9.261158e-02</td>
      <td>1.183497e+03</td>
      <td>2.456543e+00</td>
      <td>7.397005e+02</td>
      <td>3.074050e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.400000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.551000e+02</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>8.424000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>4.700000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>6.856000e+02</td>
      <td>3.000000e+00</td>
      <td>0.000000e+00</td>
      <td>4.583000e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>1.860000e+02</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>7.100000e+01</td>
      <td>1.172000e+03</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>1.910000e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.976000e+03</td>
      <td>5.000000e+00</td>
      <td>1.495000e+03</td>
      <td>7.407000e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.200000e+01</td>
      <td>3.300000e+01</td>
      <td>6.616000e+03</td>
      <td>5.300000e+01</td>
      <td>6.400000e+01</td>
      <td>8.000000e+01</td>
      <td>1.010000e+02</td>
      <td>2.170000e+03</td>
      <td>7.200000e+01</td>
      <td>2.000000e+01</td>
      <td>...</td>
      <td>3.900000e+01</td>
      <td>4.071000e+04</td>
      <td>1.800000e+01</td>
      <td>3.823000e+03</td>
      <td>1.200000e+01</td>
      <td>5.000000e+00</td>
      <td>2.578000e+04</td>
      <td>2.360000e+02</td>
      <td>2.013000e+03</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 25 columns</p>
</div>




```python
# train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4446966 entries, 0 to 4446965
    Data columns (total 29 columns):
    Id                 object
    groupId            object
    matchId            object
    assists            int64
    boosts             int64
    damageDealt        float64
    DBNOs              int64
    headshotKills      int64
    heals              int64
    killPlace          int64
    killPoints         int64
    kills              int64
    killStreaks        int64
    longestKill        float64
    matchDuration      int64
    matchType          object
    maxPlace           int64
    numGroups          int64
    rankPoints         int64
    revives            int64
    rideDistance       float64
    roadKills          int64
    swimDistance       float64
    teamKills          int64
    vehicleDestroys    int64
    walkDistance       float64
    weaponsAcquired    int64
    winPoints          int64
    winPlacePerc       float64
    dtypes: float64(6), int64(19), object(4)
    memory usage: 983.9+ MB


可以看到数据一共有4446966条，


```python
train.shape
```




    (4446965, 29)



## 数据基本处理
### 数据缺失值处理
查看目标值，我们发现有一条样本，比较特殊，其“winplaceperc”的值为NaN，也就是目标值是缺失值，

因为只有一个玩家是这样，直接进行删除处理。


```python
# 查看缺失值
train[train['winPlacePerc'].isnull()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>...</th>
      <th>revives</th>
      <th>rideDistance</th>
      <th>roadKills</th>
      <th>swimDistance</th>
      <th>teamKills</th>
      <th>vehicleDestroys</th>
      <th>walkDistance</th>
      <th>weaponsAcquired</th>
      <th>winPoints</th>
      <th>winPlacePerc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2744604</th>
      <td>f70c74418bb064</td>
      <td>12dfbede33f92b</td>
      <td>224a123c53e008</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 29 columns</p>
</div>




```python
# 删除缺失值
train.drop(2744604, inplace=True)
```


```python
train.shape
```




    (4446965, 29)



### 特征数据规范化处理
#### 查看每场比赛参加的人数
处理完缺失值之后，我们看一下每场参加的人数会有多少呢，是每次都会匹配100个人，才开始游戏吗？


```python
# 显示每场比赛参加人数
# transform的作用类似实现了一个一对多的映射功能，把统计数量映射到对应的每个样本上
count = train.groupby('matchId')['matchId'].transform('count')
```


```python
train['playersJoined'] = count
```


```python
count.count()
```




    4446965




```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>...</th>
      <th>rideDistance</th>
      <th>roadKills</th>
      <th>swimDistance</th>
      <th>teamKills</th>
      <th>vehicleDestroys</th>
      <th>walkDistance</th>
      <th>weaponsAcquired</th>
      <th>winPoints</th>
      <th>winPlacePerc</th>
      <th>playersJoined</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7f96b2f878858a</td>
      <td>4d4b580de459be</td>
      <td>a10357fd1a4a91</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>244.80</td>
      <td>1</td>
      <td>1466</td>
      <td>0.4444</td>
      <td>96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>eef90569b9d03c</td>
      <td>684d5656442f9e</td>
      <td>aeb375fc57110c</td>
      <td>0</td>
      <td>0</td>
      <td>91.47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>...</td>
      <td>0.0045</td>
      <td>0</td>
      <td>11.04</td>
      <td>0</td>
      <td>0</td>
      <td>1434.00</td>
      <td>5</td>
      <td>0</td>
      <td>0.6400</td>
      <td>91</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1eaf90ac73de72</td>
      <td>6a4a42c3245a74</td>
      <td>110163d8bb94ae</td>
      <td>1</td>
      <td>0</td>
      <td>68.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>161.80</td>
      <td>2</td>
      <td>0</td>
      <td>0.7755</td>
      <td>98</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4616d365dd2853</td>
      <td>a930a9c79cd721</td>
      <td>f1f1f4ef412d7e</td>
      <td>0</td>
      <td>0</td>
      <td>32.90</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>202.70</td>
      <td>3</td>
      <td>0</td>
      <td>0.1667</td>
      <td>91</td>
    </tr>
    <tr>
      <th>4</th>
      <td>315c96c26c9aac</td>
      <td>de04010b3458dd</td>
      <td>6dc8ff871e21e6</td>
      <td>0</td>
      <td>0</td>
      <td>100.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>...</td>
      <td>0.0000</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>49.75</td>
      <td>2</td>
      <td>0</td>
      <td>0.1875</td>
      <td>97</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>




```python
# 通过每场参加人数进行，按值升序排列
train["playersJoined"].sort_values().head()
```




    1206365    2
    2109739    2
    3956552    5
    3620228    5
    696000     5
    Name: playersJoined, dtype: int64



通过结果发现，最少的一局，竟然只有两个人，wtf!!!!


```python
# 通过绘制图像，查看每局开始人数
# 通过seaborn下的countplot方法，可以直接绘制统计过数量之后的直方图
plt.figure(figsize=(20,10))
sns.countplot(train['playersJoined'])
plt.title('playersJoined')
plt.grid()
plt.show()
```


![png](output_32_0.png)


通过观察，发现一局游戏少于75个玩家，就开始的还是比较少

同时大部分游戏都是在接近100人的时候才开始

限制每局开始人数大于等于75，再进行绘制。

猜想：把这些数据在后期加入数据处理，应该会得到的结果更加准确一些


```python
# 再次绘制每局参加人数的直方图
plt.figure(figsize=(20,10))
sns.countplot(train[train['playersJoined']>=75]['playersJoined'])
plt.title('playersJoined')
plt.grid()
plt.show()
```


![png](output_34_0.png)


#### 规范化输出部分数据

现在我们统计了“每局玩家数量”，那么我们就可以通过“每局玩家数量”来进一步考证其它特征，同时对其规范化设置

试想：一局只有70个玩家的杀敌数，和一局有100个玩家的杀敌数，应该是不可以同时比较的

可以考虑的特征值包括
 - 1.kills（杀敌数）

 - 2.damageDealt（总伤害）

 - 3.maxPlace（本局最差名次）

 - 4.matchDuration（比赛时长）


```python
# 对部分特征值进行规范化
train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)
```


```python
# 比较经过规范化的特征值和原始特征值的值
to_show = ['Id', 'kills','killsNorm','damageDealt', 'damageDealtNorm', 'maxPlace', 'maxPlaceNorm', 'matchDuration', 'matchDurationNorm']
train[to_show][0:11]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>kills</th>
      <th>killsNorm</th>
      <th>damageDealt</th>
      <th>damageDealtNorm</th>
      <th>maxPlace</th>
      <th>maxPlaceNorm</th>
      <th>matchDuration</th>
      <th>matchDurationNorm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7f96b2f878858a</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00000</td>
      <td>28</td>
      <td>29.12</td>
      <td>1306</td>
      <td>1358.24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>eef90569b9d03c</td>
      <td>0</td>
      <td>0.00</td>
      <td>91.470</td>
      <td>99.70230</td>
      <td>26</td>
      <td>28.34</td>
      <td>1777</td>
      <td>1936.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1eaf90ac73de72</td>
      <td>0</td>
      <td>0.00</td>
      <td>68.000</td>
      <td>69.36000</td>
      <td>50</td>
      <td>51.00</td>
      <td>1318</td>
      <td>1344.36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4616d365dd2853</td>
      <td>0</td>
      <td>0.00</td>
      <td>32.900</td>
      <td>35.86100</td>
      <td>31</td>
      <td>33.79</td>
      <td>1436</td>
      <td>1565.24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>315c96c26c9aac</td>
      <td>1</td>
      <td>1.03</td>
      <td>100.000</td>
      <td>103.00000</td>
      <td>97</td>
      <td>99.91</td>
      <td>1424</td>
      <td>1466.72</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ff79c12f326506</td>
      <td>1</td>
      <td>1.05</td>
      <td>100.000</td>
      <td>105.00000</td>
      <td>28</td>
      <td>29.40</td>
      <td>1395</td>
      <td>1464.75</td>
    </tr>
    <tr>
      <th>6</th>
      <td>95959be0e21ca3</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.00000</td>
      <td>28</td>
      <td>28.84</td>
      <td>1316</td>
      <td>1355.48</td>
    </tr>
    <tr>
      <th>7</th>
      <td>311b84c6ff4390</td>
      <td>0</td>
      <td>0.00</td>
      <td>8.538</td>
      <td>8.87952</td>
      <td>96</td>
      <td>99.84</td>
      <td>1967</td>
      <td>2045.68</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1a68204ccf9891</td>
      <td>0</td>
      <td>0.00</td>
      <td>51.600</td>
      <td>53.14800</td>
      <td>28</td>
      <td>28.84</td>
      <td>1375</td>
      <td>1416.25</td>
    </tr>
    <tr>
      <th>9</th>
      <td>e5bb5a43587253</td>
      <td>0</td>
      <td>0.00</td>
      <td>37.270</td>
      <td>38.38810</td>
      <td>29</td>
      <td>29.87</td>
      <td>1930</td>
      <td>1987.90</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2b574d43972813</td>
      <td>0</td>
      <td>0.00</td>
      <td>28.380</td>
      <td>28.66380</td>
      <td>29</td>
      <td>29.29</td>
      <td>1811</td>
      <td>1829.11</td>
    </tr>
  </tbody>
</table>
</div>



### 部分变量合成
此处我们把特征：heals(使用治疗药品数量)和boosts(能量、道具使用数量)合并成一个新的变量，命名：”healsandboosts“， 这是一个探索性过程，最后结果不一定有用，如果没有实际用处，最后再把它删除。


```python
# 创建新变量“healsandboosts”
train['healsandboosts'] = train['heals'] + train['boosts']
```


```python
train[["heals", "boosts", "healsandboosts"]].tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>heals</th>
      <th>boosts</th>
      <th>healsandboosts</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4446961</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4446962</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4446963</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4446964</th>
      <td>2</td>
      <td>4</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4446965</th>
      <td>1</td>
      <td>2</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>



### 异常值处理
#### 异常值处理：删除有击杀，但是完全没有移动的玩家
异常数据处理：

一些行中的数据统计出来的结果非常反常规，那么这些玩家肯定有问题，为了训练模型的准确性，我们会把这些异常数据剔除

通过以下操作，识别出玩家在游戏中有击杀数，但是全局没有移动；

这类型玩家肯定是存在异常情况（挂**），我们把这些玩家删除。


```python
# 创建新变量，统计玩家移动距离
train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
# 创建新变量，统计玩家是否在游戏中，有击杀，但是没有移动，如果是返回True, 否则返回false
train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))
```


```python
train["killsWithoutMoving"].head()
```




    0    False
    1    False
    2    False
    3    False
    4    False
    Name: killsWithoutMoving, dtype: bool




```python
train["killsWithoutMoving"].describe()
```




    count     4446965
    unique          2
    top         False
    freq      4445430
    Name: killsWithoutMoving, dtype: object




```python
# 检查是否存在有击杀但是没有移动的数据
train[train['killsWithoutMoving'] == True].shape
```




    (1535, 37)




```python
train[train['killsWithoutMoving'] == True].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>...</th>
      <th>winPoints</th>
      <th>winPlacePerc</th>
      <th>playersJoined</th>
      <th>killsNorm</th>
      <th>damageDealtNorm</th>
      <th>maxPlaceNorm</th>
      <th>matchDurationNorm</th>
      <th>healsandboosts</th>
      <th>totalDistance</th>
      <th>killsWithoutMoving</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1824</th>
      <td>b538d514ef2476</td>
      <td>0eb2ce2f43f9d6</td>
      <td>35e7d750e442e2</td>
      <td>0</td>
      <td>0</td>
      <td>593.0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>18</td>
      <td>...</td>
      <td>0</td>
      <td>0.8571</td>
      <td>58</td>
      <td>8.52</td>
      <td>842.060</td>
      <td>21.30</td>
      <td>842.06</td>
      <td>3</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>6673</th>
      <td>6d3a61da07b7cb</td>
      <td>2d8119b1544f87</td>
      <td>904cecf36217df</td>
      <td>2</td>
      <td>0</td>
      <td>346.6</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>33</td>
      <td>...</td>
      <td>0</td>
      <td>0.6000</td>
      <td>42</td>
      <td>4.74</td>
      <td>547.628</td>
      <td>17.38</td>
      <td>2834.52</td>
      <td>6</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>11892</th>
      <td>550398a8f33db7</td>
      <td>c3fd0e2abab0af</td>
      <td>db6f6d1f0d4904</td>
      <td>2</td>
      <td>0</td>
      <td>1750.0</td>
      <td>0</td>
      <td>4</td>
      <td>5</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0.8947</td>
      <td>21</td>
      <td>35.80</td>
      <td>3132.500</td>
      <td>35.80</td>
      <td>1607.42</td>
      <td>5</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>14631</th>
      <td>58d690ee461e9d</td>
      <td>ea5b6630b33d67</td>
      <td>dbf34301df5e53</td>
      <td>0</td>
      <td>0</td>
      <td>157.8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>69</td>
      <td>...</td>
      <td>1500</td>
      <td>0.0000</td>
      <td>73</td>
      <td>1.27</td>
      <td>200.406</td>
      <td>24.13</td>
      <td>1014.73</td>
      <td>0</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>15591</th>
      <td>49b61fc963d632</td>
      <td>0f5c5f19d9cc21</td>
      <td>904cecf36217df</td>
      <td>0</td>
      <td>0</td>
      <td>100.0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>37</td>
      <td>...</td>
      <td>0</td>
      <td>0.3000</td>
      <td>42</td>
      <td>1.58</td>
      <td>158.000</td>
      <td>17.38</td>
      <td>2834.52</td>
      <td>0</td>
      <td>0.0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
# 删除这些数据
train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
```

#### 异常值处理：删除驾车杀敌数异常的数据


```python
# 查看载具杀敌数超过十个的玩家
train[train['roadKills'] > 10]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>...</th>
      <th>winPoints</th>
      <th>winPlacePerc</th>
      <th>playersJoined</th>
      <th>killsNorm</th>
      <th>damageDealtNorm</th>
      <th>maxPlaceNorm</th>
      <th>matchDurationNorm</th>
      <th>healsandboosts</th>
      <th>totalDistance</th>
      <th>killsWithoutMoving</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2733926</th>
      <td>c3e444f7d1289f</td>
      <td>489dd6d1f2b3bb</td>
      <td>4797482205aaa4</td>
      <td>0</td>
      <td>0</td>
      <td>1246.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1371</td>
      <td>0.4286</td>
      <td>92</td>
      <td>15.12</td>
      <td>1345.68</td>
      <td>99.36</td>
      <td>1572.48</td>
      <td>0</td>
      <td>1282.302</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2767999</th>
      <td>34193085975338</td>
      <td>bd7d50fa305700</td>
      <td>a22354d036b3d6</td>
      <td>0</td>
      <td>0</td>
      <td>1102.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1533</td>
      <td>0.4713</td>
      <td>88</td>
      <td>12.32</td>
      <td>1234.24</td>
      <td>98.56</td>
      <td>2179.52</td>
      <td>0</td>
      <td>4934.600</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2890740</th>
      <td>a3438934e3e535</td>
      <td>1081c315a80d14</td>
      <td>fe744430ac0070</td>
      <td>0</td>
      <td>8</td>
      <td>2074.0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>...</td>
      <td>1568</td>
      <td>1.0000</td>
      <td>38</td>
      <td>32.40</td>
      <td>3359.88</td>
      <td>61.56</td>
      <td>3191.40</td>
      <td>19</td>
      <td>5876.000</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3524413</th>
      <td>9d9d044f81de72</td>
      <td>8be97e1ba792e3</td>
      <td>859e2c2db5b125</td>
      <td>0</td>
      <td>3</td>
      <td>1866.0</td>
      <td>0</td>
      <td>5</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>1606</td>
      <td>0.9398</td>
      <td>84</td>
      <td>20.88</td>
      <td>2164.56</td>
      <td>97.44</td>
      <td>2233.00</td>
      <td>10</td>
      <td>7853.000</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 37 columns</p>
</div>




```python
# 删除这些数据
train.drop(train[train['roadKills'] > 10].index, inplace=True)
```


```python
train.shape
```




    (4445426, 37)



#### 异常值处理：删除玩家在一局中杀敌数超过30人的数据


```python
# 首先绘制玩家杀敌数的条形图
plt.figure(figsize=(10,4))
sns.countplot(data=train, x=train['kills']).set_title('Kills')
plt.show()
```


![png](output_53_0.png)



```python
train[train['kills'] > 30].shape
```




    (95, 37)




```python
train[train['kills'] > 30].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>...</th>
      <th>winPoints</th>
      <th>winPlacePerc</th>
      <th>playersJoined</th>
      <th>killsNorm</th>
      <th>damageDealtNorm</th>
      <th>maxPlaceNorm</th>
      <th>matchDurationNorm</th>
      <th>healsandboosts</th>
      <th>totalDistance</th>
      <th>killsWithoutMoving</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>57978</th>
      <td>9d8253e21ccbbd</td>
      <td>ef7135ed856cd8</td>
      <td>37f05e2a01015f</td>
      <td>9</td>
      <td>0</td>
      <td>3725.0</td>
      <td>0</td>
      <td>7</td>
      <td>0</td>
      <td>2</td>
      <td>...</td>
      <td>1500</td>
      <td>0.8571</td>
      <td>16</td>
      <td>64.40</td>
      <td>6854.00</td>
      <td>14.72</td>
      <td>3308.32</td>
      <td>0</td>
      <td>48.82</td>
      <td>False</td>
    </tr>
    <tr>
      <th>87793</th>
      <td>45f76442384931</td>
      <td>b3627758941d34</td>
      <td>37f05e2a01015f</td>
      <td>8</td>
      <td>0</td>
      <td>3087.0</td>
      <td>0</td>
      <td>8</td>
      <td>27</td>
      <td>3</td>
      <td>...</td>
      <td>1500</td>
      <td>1.0000</td>
      <td>16</td>
      <td>57.04</td>
      <td>5680.08</td>
      <td>14.72</td>
      <td>3308.32</td>
      <td>27</td>
      <td>780.70</td>
      <td>False</td>
    </tr>
    <tr>
      <th>156599</th>
      <td>746aa7eabf7c86</td>
      <td>5723e7d8250da3</td>
      <td>f900de1ec39fa5</td>
      <td>21</td>
      <td>0</td>
      <td>5479.0</td>
      <td>0</td>
      <td>12</td>
      <td>7</td>
      <td>4</td>
      <td>...</td>
      <td>0</td>
      <td>0.7000</td>
      <td>11</td>
      <td>90.72</td>
      <td>10355.31</td>
      <td>20.79</td>
      <td>3398.22</td>
      <td>7</td>
      <td>23.71</td>
      <td>False</td>
    </tr>
    <tr>
      <th>160254</th>
      <td>15622257cb44e2</td>
      <td>1a513eeecfe724</td>
      <td>db413c7c48292c</td>
      <td>1</td>
      <td>0</td>
      <td>4033.0</td>
      <td>0</td>
      <td>40</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1500</td>
      <td>1.0000</td>
      <td>62</td>
      <td>57.96</td>
      <td>5565.54</td>
      <td>11.04</td>
      <td>1164.72</td>
      <td>0</td>
      <td>718.30</td>
      <td>False</td>
    </tr>
    <tr>
      <th>180189</th>
      <td>1355613d43e2d0</td>
      <td>f863cd38c61dbf</td>
      <td>39c442628f5df5</td>
      <td>5</td>
      <td>0</td>
      <td>3171.0</td>
      <td>0</td>
      <td>6</td>
      <td>15</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>1.0000</td>
      <td>11</td>
      <td>66.15</td>
      <td>5993.19</td>
      <td>17.01</td>
      <td>3394.44</td>
      <td>15</td>
      <td>71.51</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 37 columns</p>
</div>




```python
# 异常数据删除
train.drop(train[train['kills'] > 30].index, inplace=True)
```

#### 异常值处理：删除爆头率异常数据
如果一个玩家的击杀爆头率过高，也说明其有问题


```python
# 创建变量爆头率
train['headshot_rate'] = train['headshotKills'] / train['kills']
train['headshot_rate'] = train['headshot_rate'].fillna(0)
```


```python
train["headshot_rate"].tail()
```




    4446961    0.0
    4446962    0.0
    4446963    0.0
    4446964    0.5
    4446965    0.0
    Name: headshot_rate, dtype: float64




```python
# 绘制爆头率图像
plt.figure(figsize=(12,4))
sns.distplot(train['headshot_rate'], bins=10)
plt.show()
```


![png](output_60_0.png)



```python
train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].shape
```




    (24, 38)




```python
train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>...</th>
      <th>winPlacePerc</th>
      <th>playersJoined</th>
      <th>killsNorm</th>
      <th>damageDealtNorm</th>
      <th>maxPlaceNorm</th>
      <th>matchDurationNorm</th>
      <th>healsandboosts</th>
      <th>totalDistance</th>
      <th>killsWithoutMoving</th>
      <th>headshot_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>281570</th>
      <td>ab9d7168570927</td>
      <td>add05ebde0214c</td>
      <td>e016a873339c7b</td>
      <td>2</td>
      <td>3</td>
      <td>1212.0</td>
      <td>8</td>
      <td>10</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0.8462</td>
      <td>93</td>
      <td>10.70</td>
      <td>1296.84</td>
      <td>28.89</td>
      <td>1522.61</td>
      <td>3</td>
      <td>2939.0</td>
      <td>False</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>346124</th>
      <td>044d18fc42fc75</td>
      <td>fc1dbc2df6a887</td>
      <td>628107d4c41084</td>
      <td>3</td>
      <td>5</td>
      <td>1620.0</td>
      <td>13</td>
      <td>11</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>1.0000</td>
      <td>96</td>
      <td>11.44</td>
      <td>1684.80</td>
      <td>28.08</td>
      <td>1796.08</td>
      <td>8</td>
      <td>8142.0</td>
      <td>False</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>871244</th>
      <td>e668a25f5488e3</td>
      <td>5ba8feabfb2a23</td>
      <td>f6e6581e03ba4f</td>
      <td>0</td>
      <td>4</td>
      <td>1365.0</td>
      <td>9</td>
      <td>13</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>1.0000</td>
      <td>98</td>
      <td>13.26</td>
      <td>1392.30</td>
      <td>27.54</td>
      <td>1280.10</td>
      <td>4</td>
      <td>2105.0</td>
      <td>False</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>908815</th>
      <td>566d8218b705aa</td>
      <td>a9b056478d71b2</td>
      <td>3a41552d553583</td>
      <td>2</td>
      <td>5</td>
      <td>1535.0</td>
      <td>10</td>
      <td>10</td>
      <td>3</td>
      <td>1</td>
      <td>...</td>
      <td>0.9630</td>
      <td>95</td>
      <td>10.50</td>
      <td>1611.75</td>
      <td>29.40</td>
      <td>1929.90</td>
      <td>8</td>
      <td>7948.0</td>
      <td>False</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>963463</th>
      <td>1bd6fd288df4f0</td>
      <td>90584ffa22fe15</td>
      <td>ba2de992ec7bb8</td>
      <td>2</td>
      <td>6</td>
      <td>1355.0</td>
      <td>12</td>
      <td>10</td>
      <td>2</td>
      <td>1</td>
      <td>...</td>
      <td>1.0000</td>
      <td>96</td>
      <td>10.40</td>
      <td>1409.20</td>
      <td>28.08</td>
      <td>1473.68</td>
      <td>8</td>
      <td>3476.0</td>
      <td>False</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
train.drop(train[(train['headshot_rate'] == 1) & (train['kills'] > 9)].index, inplace=True)
```

#### 异常值处理：删除最远杀敌距离异常数据


```python
# 绘制图像
plt.figure(figsize=(12,4))
sns.distplot(train['longestKill'], bins=10)
plt.show()
```


![png](output_65_0.png)



```python
# 找出最远杀敌距离大于等于1km的玩家
train[train['longestKill'] >= 1000].shape
```




    (20, 38)




```python
train[train['longestKill'] >= 1000]["longestKill"].head()
```




    202281    1000.0
    240005    1004.0
    324313    1026.0
    656553    1000.0
    803632    1075.0
    Name: longestKill, dtype: float64




```python
train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
```


```python
train.shape
```




    (4445287, 38)



#### 异常值处理：删除关于运动距离的异常值


```python
# 距离整体描述
train[['walkDistance', 'rideDistance', 'swimDistance', 'totalDistance']].describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>walkDistance</th>
      <th>rideDistance</th>
      <th>swimDistance</th>
      <th>totalDistance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.445287e+06</td>
      <td>4.445287e+06</td>
      <td>4.445287e+06</td>
      <td>4.445287e+06</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.154619e+03</td>
      <td>6.063215e+02</td>
      <td>4.510898e+00</td>
      <td>1.765451e+03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.183508e+03</td>
      <td>1.498562e+03</td>
      <td>3.050738e+01</td>
      <td>2.183248e+03</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.554000e+02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>1.584000e+02</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.863000e+02</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>7.892500e+02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.977000e+03</td>
      <td>2.566000e-01</td>
      <td>0.000000e+00</td>
      <td>2.729000e+03</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.578000e+04</td>
      <td>4.071000e+04</td>
      <td>3.823000e+03</td>
      <td>4.127010e+04</td>
    </tr>
  </tbody>
</table>
</div>



##### a）行走距离处理


```python
plt.figure(figsize=(12,4))
sns.distplot(train['walkDistance'], bins=10)
plt.show()
```


![png](output_73_0.png)



```python
train[train['walkDistance'] >= 10000].shape
```




    (219, 38)




```python
train[train['walkDistance'] >= 10000].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>...</th>
      <th>winPlacePerc</th>
      <th>playersJoined</th>
      <th>killsNorm</th>
      <th>damageDealtNorm</th>
      <th>maxPlaceNorm</th>
      <th>matchDurationNorm</th>
      <th>healsandboosts</th>
      <th>totalDistance</th>
      <th>killsWithoutMoving</th>
      <th>headshot_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>23026</th>
      <td>8a6562381dd83f</td>
      <td>23e638cd6eaf77</td>
      <td>b0a804a610e9b0</td>
      <td>0</td>
      <td>1</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>...</td>
      <td>0.8163</td>
      <td>99</td>
      <td>0.00</td>
      <td>0.0000</td>
      <td>99.99</td>
      <td>1925.06</td>
      <td>1</td>
      <td>13540.3032</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>34344</th>
      <td>5a591ecc957393</td>
      <td>6717370b51c247</td>
      <td>a15d93e7165b05</td>
      <td>0</td>
      <td>3</td>
      <td>23.22</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>34</td>
      <td>...</td>
      <td>0.9474</td>
      <td>65</td>
      <td>0.00</td>
      <td>31.3470</td>
      <td>27.00</td>
      <td>2668.95</td>
      <td>4</td>
      <td>10070.9073</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>49312</th>
      <td>582685f487f0b4</td>
      <td>338112cd12f1e7</td>
      <td>d0afbf5c3a6dc9</td>
      <td>0</td>
      <td>4</td>
      <td>117.20</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>24</td>
      <td>...</td>
      <td>0.9130</td>
      <td>94</td>
      <td>1.06</td>
      <td>124.2320</td>
      <td>49.82</td>
      <td>2323.52</td>
      <td>5</td>
      <td>12446.7588</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>68590</th>
      <td>8c0d9dd0b4463c</td>
      <td>c963553dc937e9</td>
      <td>926681ea721a47</td>
      <td>0</td>
      <td>1</td>
      <td>32.34</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>46</td>
      <td>...</td>
      <td>0.8333</td>
      <td>96</td>
      <td>0.00</td>
      <td>33.6336</td>
      <td>50.96</td>
      <td>1909.44</td>
      <td>2</td>
      <td>12483.6200</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>94400</th>
      <td>d441bebd01db61</td>
      <td>7e179b3366adb8</td>
      <td>923b57b8b834cc</td>
      <td>1</td>
      <td>1</td>
      <td>73.08</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>27</td>
      <td>...</td>
      <td>0.8194</td>
      <td>73</td>
      <td>0.00</td>
      <td>92.8116</td>
      <td>92.71</td>
      <td>2293.62</td>
      <td>4</td>
      <td>11490.6300</td>
      <td>False</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
train.drop(train[train['walkDistance'] >= 10000].index, inplace=True)
```

#####  b）载具行驶距离处理


```python
plt.figure(figsize=(12,4))
sns.distplot(train['rideDistance'], bins=10)
plt.show()
```


![png](output_78_0.png)



```python
train[train['rideDistance'] >= 20000].shape
```




    (150, 38)




```python
train[train['rideDistance'] >= 20000].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>groupId</th>
      <th>matchId</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>...</th>
      <th>winPlacePerc</th>
      <th>playersJoined</th>
      <th>killsNorm</th>
      <th>damageDealtNorm</th>
      <th>maxPlaceNorm</th>
      <th>matchDurationNorm</th>
      <th>healsandboosts</th>
      <th>totalDistance</th>
      <th>killsWithoutMoving</th>
      <th>headshot_rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28588</th>
      <td>6260f7c49dc16f</td>
      <td>b24589f02eedd7</td>
      <td>6ebea3b4f55b4a</td>
      <td>0</td>
      <td>0</td>
      <td>99.2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>30</td>
      <td>...</td>
      <td>0.6421</td>
      <td>96</td>
      <td>1.04</td>
      <td>103.168</td>
      <td>99.84</td>
      <td>1969.76</td>
      <td>1</td>
      <td>26306.6</td>
      <td>False</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>63015</th>
      <td>adb7dae4d0c10a</td>
      <td>8ede98a241f30a</td>
      <td>8b36eac66378e4</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>55</td>
      <td>...</td>
      <td>0.5376</td>
      <td>94</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>99.64</td>
      <td>2004.46</td>
      <td>0</td>
      <td>22065.4</td>
      <td>False</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>70507</th>
      <td>ca6fa339064d67</td>
      <td>f7bb2e30c3461f</td>
      <td>3bfd8d66edbeff</td>
      <td>0</td>
      <td>0</td>
      <td>100.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>26</td>
      <td>...</td>
      <td>0.8878</td>
      <td>99</td>
      <td>1.01</td>
      <td>101.000</td>
      <td>99.99</td>
      <td>1947.28</td>
      <td>0</td>
      <td>28917.5</td>
      <td>False</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>72763</th>
      <td>198e5894e68ff4</td>
      <td>ccf47c82abb11f</td>
      <td>d92bf8e696b61d</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>46</td>
      <td>...</td>
      <td>0.7917</td>
      <td>97</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>99.91</td>
      <td>1861.21</td>
      <td>0</td>
      <td>21197.2</td>
      <td>False</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>95276</th>
      <td>c3fabfce7589ae</td>
      <td>15529e25aa4a74</td>
      <td>d055504340e5f4</td>
      <td>0</td>
      <td>7</td>
      <td>778.2</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>...</td>
      <td>0.9785</td>
      <td>94</td>
      <td>7.42</td>
      <td>824.892</td>
      <td>99.64</td>
      <td>1986.44</td>
      <td>9</td>
      <td>26733.2</td>
      <td>False</td>
      <td>0.142857</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 38 columns</p>
</div>




```python
train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
```

##### c）游泳距离处理


```python
plt.figure(figsize=(12,4))
sns.distplot(train['swimDistance'], bins=10)
plt.show()
```


![png](output_83_0.png)



```python
train[train['swimDistance'] >= 2000].shape
```




    (12, 38)




```python
train[train['swimDistance'] >= 2000][["swimDistance"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>swimDistance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>177973</th>
      <td>2295.0</td>
    </tr>
    <tr>
      <th>274258</th>
      <td>2148.0</td>
    </tr>
    <tr>
      <th>1005337</th>
      <td>2718.0</td>
    </tr>
    <tr>
      <th>1195818</th>
      <td>2668.0</td>
    </tr>
    <tr>
      <th>1227362</th>
      <td>3823.0</td>
    </tr>
    <tr>
      <th>1889163</th>
      <td>2484.0</td>
    </tr>
    <tr>
      <th>2065940</th>
      <td>3514.0</td>
    </tr>
    <tr>
      <th>2327586</th>
      <td>2387.0</td>
    </tr>
    <tr>
      <th>2784855</th>
      <td>2206.0</td>
    </tr>
    <tr>
      <th>3359439</th>
      <td>2338.0</td>
    </tr>
    <tr>
      <th>3513522</th>
      <td>2124.0</td>
    </tr>
    <tr>
      <th>4132225</th>
      <td>2382.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
```

#### 异常值处理：武器收集异常值处理


```python
plt.figure(figsize=(12,4))
sns.distplot(train['weaponsAcquired'], bins=100)
plt.show()
```


![png](output_88_0.png)



```python
train[train['weaponsAcquired'] >= 80].shape
```




    (19, 38)




```python
train[train['weaponsAcquired'] >= 80][['weaponsAcquired']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>weaponsAcquired</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>233643</th>
      <td>128</td>
    </tr>
    <tr>
      <th>588387</th>
      <td>80</td>
    </tr>
    <tr>
      <th>1437471</th>
      <td>102</td>
    </tr>
    <tr>
      <th>1449293</th>
      <td>95</td>
    </tr>
    <tr>
      <th>1592744</th>
      <td>94</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
```

#### 异常值处理：删除使用治疗药品数量异常值


```python
plt.figure(figsize=(12,4))
sns.distplot(train['heals'], bins=10)
plt.show()
```


![png](output_93_0.png)



```python
train[train['heals'] >= 40].shape
```




    (135, 38)




```python
train[train['heals'] >= 40][["heals"]].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>heals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>18405</th>
      <td>47</td>
    </tr>
    <tr>
      <th>54463</th>
      <td>43</td>
    </tr>
    <tr>
      <th>126439</th>
      <td>52</td>
    </tr>
    <tr>
      <th>259351</th>
      <td>42</td>
    </tr>
    <tr>
      <th>268747</th>
      <td>48</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.drop(train[train['heals'] >= 40].index, inplace=True)
```


```python
train.shape
```




    (4444752, 38)



### 类别型数据处理
#### 比赛类型one-hot处理


```python
# 关于比赛类型，共有16种方式
train['matchType'].unique()
```




    array(['squad-fpp', 'duo', 'solo-fpp', 'squad', 'duo-fpp', 'solo',
           'normal-squad-fpp', 'crashfpp', 'flaretpp', 'normal-solo-fpp',
           'flarefpp', 'normal-duo-fpp', 'normal-duo', 'normal-squad',
           'crashtpp', 'normal-solo'], dtype=object)




```python
# 对matchType进行one_hot编码
# 通过在后面添加的方式,实现,赋值并不是替换
train = pd.get_dummies(train, columns=['matchType'])
```


```python
train.shape
```




    (4444752, 53)




```python
# 通过正则匹配查看具体内容
matchType_encoding = train.filter(regex='matchType')
matchType_encoding.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>matchType_crashfpp</th>
      <th>matchType_crashtpp</th>
      <th>matchType_duo</th>
      <th>matchType_duo-fpp</th>
      <th>matchType_flarefpp</th>
      <th>matchType_flaretpp</th>
      <th>matchType_normal-duo</th>
      <th>matchType_normal-duo-fpp</th>
      <th>matchType_normal-solo</th>
      <th>matchType_normal-solo-fpp</th>
      <th>matchType_normal-squad</th>
      <th>matchType_normal-squad-fpp</th>
      <th>matchType_solo</th>
      <th>matchType_solo-fpp</th>
      <th>matchType_squad</th>
      <th>matchType_squad-fpp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



#### 对groupId,matchId等数据进行处理
关于groupId,matchId这类型数据，也是类别型数据。但是它们的数据量特别多，如果你使用one-hot编码，无异于自杀。

在这儿我们把它们变成用数字统计的类别型数据依旧不影响我们正常使用。


```python
# 把groupId 和 match Id 转换成类别类型 categorical types
# 就是把一堆不怎么好识别的内容转换成数字

# 转换group_id
train["groupId"].head()
```




    0    4d4b580de459be
    1    684d5656442f9e
    2    6a4a42c3245a74
    3    a930a9c79cd721
    4    de04010b3458dd
    Name: groupId, dtype: object




```python
train['groupId'] = train['groupId'].astype('category')
```


```python
train["groupId"].head()
```




    0    4d4b580de459be
    1    684d5656442f9e
    2    6a4a42c3245a74
    3    a930a9c79cd721
    4    de04010b3458dd
    Name: groupId, dtype: category
    Categories (2026153, object): [00000c08b5be36, 00000d1cbbc340, 000025a09dd1d7, 000038ec4dff53, ..., fffff305a0133d, fffff32bc7eab9, fffff7edfc4050, fffff98178ef52]




```python
train["groupId_cat"] = train["groupId"].cat.codes
```


```python
train["groupId_cat"].head()
```




    0     613591
    1     827580
    2     843271
    3    1340070
    4    1757334
    Name: groupId_cat, dtype: int32




```python
# 转换match_id
train['matchId'] = train['matchId'].astype('category')

train['matchId_cat'] = train['matchId'].cat.codes

```


```python
# 删除之前列
train.drop(['groupId', 'matchId'], axis=1, inplace=True)

# 查看新产生列
train[['groupId_cat', 'matchId_cat']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>groupId_cat</th>
      <th>matchId_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>613591</td>
      <td>30085</td>
    </tr>
    <tr>
      <th>1</th>
      <td>827580</td>
      <td>32751</td>
    </tr>
    <tr>
      <th>2</th>
      <td>843271</td>
      <td>3143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1340070</td>
      <td>45260</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1757334</td>
      <td>20531</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>killPoints</th>
      <th>kills</th>
      <th>...</th>
      <th>matchType_normal-solo</th>
      <th>matchType_normal-solo-fpp</th>
      <th>matchType_normal-squad</th>
      <th>matchType_normal-squad-fpp</th>
      <th>matchType_solo</th>
      <th>matchType_solo-fpp</th>
      <th>matchType_squad</th>
      <th>matchType_squad-fpp</th>
      <th>groupId_cat</th>
      <th>matchId_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7f96b2f878858a</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>60</td>
      <td>1241</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>613591</td>
      <td>30085</td>
    </tr>
    <tr>
      <th>1</th>
      <td>eef90569b9d03c</td>
      <td>0</td>
      <td>0</td>
      <td>91.47</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>57</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>827580</td>
      <td>32751</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1eaf90ac73de72</td>
      <td>1</td>
      <td>0</td>
      <td>68.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>47</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>843271</td>
      <td>3143</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4616d365dd2853</td>
      <td>0</td>
      <td>0</td>
      <td>32.90</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>75</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1340070</td>
      <td>45260</td>
    </tr>
    <tr>
      <th>4</th>
      <td>315c96c26c9aac</td>
      <td>0</td>
      <td>0</td>
      <td>100.00</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>45</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1757334</td>
      <td>20531</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 53 columns</p>
</div>



### 数据截取
#### 取部分数据进行使用（1000000）


```python
# 取前100万条数据，进行训练
sample = 1000000
df_sample = train.sample(sample)
```


```python
df_sample.shape
```




    (1000000, 53)



### 确定特征值和目标值


```python
# 确定特征值和目标值
df = df_sample.drop(["winPlacePerc", "Id"], axis=1) #all columns except target

y = df_sample['winPlacePerc'] # Only target variable
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>assists</th>
      <th>boosts</th>
      <th>damageDealt</th>
      <th>DBNOs</th>
      <th>headshotKills</th>
      <th>heals</th>
      <th>killPlace</th>
      <th>killPoints</th>
      <th>kills</th>
      <th>killStreaks</th>
      <th>...</th>
      <th>matchType_normal-solo</th>
      <th>matchType_normal-solo-fpp</th>
      <th>matchType_normal-squad</th>
      <th>matchType_normal-squad-fpp</th>
      <th>matchType_solo</th>
      <th>matchType_solo-fpp</th>
      <th>matchType_squad</th>
      <th>matchType_squad-fpp</th>
      <th>groupId_cat</th>
      <th>matchId_cat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>565181</th>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>29</td>
      <td>1210</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>784273</td>
      <td>45231</td>
    </tr>
    <tr>
      <th>1278768</th>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>70</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>724178</td>
      <td>24323</td>
    </tr>
    <tr>
      <th>884983</th>
      <td>1</td>
      <td>4</td>
      <td>170.1</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>23</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>978249</td>
      <td>12809</td>
    </tr>
    <tr>
      <th>3040251</th>
      <td>0</td>
      <td>2</td>
      <td>273.2</td>
      <td>1</td>
      <td>0</td>
      <td>5</td>
      <td>22</td>
      <td>1173</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>184567</td>
      <td>31273</td>
    </tr>
    <tr>
      <th>2285385</th>
      <td>1</td>
      <td>0</td>
      <td>178.0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>44</td>
      <td>1245</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>889319</td>
      <td>3893</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 51 columns</p>
</div>




```python
y.head()
```




    565181     0.5217
    1278768    0.2527
    884983     0.9231
    3040251    0.8889
    2285385    0.0417
    Name: winPlacePerc, dtype: float64




```python
print(df.shape, y.shape)
```

    (1000000, 51) (1000000,)


### 分割训练集和验证集


```python
# 自定义函数，分割训练集和验证集
def split_vals(a, n : int): 
    # ps: n:int 是一种新的定义函数方式，告诉你这个n,传入应该是int类型，但不是强制的
    return a[:n].copy(), a[n:].copy()
val_perc = 0.12 # % to use for validation set
n_valid = int(val_perc * sample) 
n_trn = len(df)-n_valid

# 分割数据集
raw_train, raw_valid = split_vals(df_sample, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

# 检查数据集维度
print('Sample train shape: ', X_train.shape, 
      '\nSample target shape: ', y_train.shape, 
      '\nSample validation shape: ', X_valid.shape)
```

    Sample train shape:  (880000, 51) 
    Sample target shape:  (880000,) 
    Sample validation shape:  (120000, 51)


## 机器学习（模型训练）和评估


```python
# 导入需要训练和评估api
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
```

### 初步使用随机森林进行模型训练


```python
# 模型训练
m1 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',
                          n_jobs=-1)
# n_jobs=-1 表示训练的时候，并行数和cpu的核数一样，如果传入具体的值，表示用几个核去跑

m1.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='sqrt', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=3, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=-1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)




```python
y_pre = m1.predict(X_valid)
m1.score(X_valid, y_valid)
```




    0.9211261114580913




```python
mean_absolute_error(y_true=y_valid, y_pred=y_pre)
```




    0.06134628883773538



经过第一次计算，得出准确率为：0.92， mae=0.06

### 再次使用随机森林，进行模型训练
减少特征值，提高模型训练效率


```python
# 查看特征值在当前模型中的重要程度
m1.feature_importances_
```




    array([6.35095577e-03, 7.76349587e-02, 2.44946259e-02, 1.76567724e-03,
           1.27014454e-03, 4.32254115e-02, 1.67783107e-01, 1.99200052e-03,
           6.39609344e-03, 2.88255931e-03, 9.50856157e-03, 1.09690695e-02,
           5.87901976e-03, 7.54892226e-03, 3.48302903e-03, 8.02457958e-04,
           1.55288860e-02, 3.31198839e-05, 1.56341614e-03, 1.24483687e-04,
           6.44617997e-05, 2.90355609e-01, 6.49551068e-02, 2.31528973e-03,
           6.22359074e-03, 1.02285994e-02, 7.81993834e-03, 7.00519528e-03,
           1.18603355e-02, 2.90628266e-02, 1.65072310e-01, 0.00000000e+00,
           2.94266078e-03, 5.59029567e-05, 6.05200130e-07, 2.16466045e-04,
           4.99625694e-04, 1.21800071e-07, 1.91392771e-06, 5.08221519e-07,
           6.59823220e-05, 5.78083365e-07, 1.77813484e-05, 7.06918802e-07,
           2.98436246e-04, 2.56781611e-04, 1.18436859e-03, 1.14809540e-03,
           9.20847747e-04, 4.10576981e-03, 4.08308467e-03])




```python
imp_df = pd.DataFrame({"cols":df.columns, "imp":m1.feature_importances_})
```


```python
imp_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>assists</td>
      <td>0.006351</td>
    </tr>
    <tr>
      <th>1</th>
      <td>boosts</td>
      <td>0.077635</td>
    </tr>
    <tr>
      <th>2</th>
      <td>damageDealt</td>
      <td>0.024495</td>
    </tr>
    <tr>
      <th>3</th>
      <td>DBNOs</td>
      <td>0.001766</td>
    </tr>
    <tr>
      <th>4</th>
      <td>headshotKills</td>
      <td>0.001270</td>
    </tr>
  </tbody>
</table>
</div>




```python
imp_df = imp_df.sort_values("imp", ascending=False)
```


```python
imp_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>cols</th>
      <th>imp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>walkDistance</td>
      <td>0.290356</td>
    </tr>
    <tr>
      <th>6</th>
      <td>killPlace</td>
      <td>0.167783</td>
    </tr>
    <tr>
      <th>30</th>
      <td>totalDistance</td>
      <td>0.165072</td>
    </tr>
    <tr>
      <th>1</th>
      <td>boosts</td>
      <td>0.077635</td>
    </tr>
    <tr>
      <th>22</th>
      <td>weaponsAcquired</td>
      <td>0.064955</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot a feature importance graph for the 20 most important features
# 绘制特征重要性程度图，仅展示排名前二十的特征
plot_fea = imp_df[:20].plot('cols', 'imp', figsize=(14,6), legend=False, kind = 'barh')
plot_fea
```




    <matplotlib.axes._subplots.AxesSubplot at 0x118d9ea90>




![png](output_135_1.png)



```python
# 保留比较重要的特征
to_keep = imp_df[imp_df.imp>0.005].cols
print('Significant features: ', len(to_keep))
to_keep
```

    Significant features:  20





    21         walkDistance
    6             killPlace
    30        totalDistance
    1                boosts
    22      weaponsAcquired
    5                 heals
    29       healsandboosts
    2           damageDealt
    16         rideDistance
    28    matchDurationNorm
    11        matchDuration
    25            killsNorm
    10          longestKill
    26      damageDealtNorm
    13            numGroups
    27         maxPlaceNorm
    8                 kills
    0               assists
    24        playersJoined
    12             maxPlace
    Name: cols, dtype: object




```python
# 由这些比较重要的特征值，生成新的df
df[to_keep].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>walkDistance</th>
      <th>killPlace</th>
      <th>totalDistance</th>
      <th>boosts</th>
      <th>weaponsAcquired</th>
      <th>heals</th>
      <th>healsandboosts</th>
      <th>damageDealt</th>
      <th>rideDistance</th>
      <th>matchDurationNorm</th>
      <th>matchDuration</th>
      <th>killsNorm</th>
      <th>longestKill</th>
      <th>damageDealtNorm</th>
      <th>numGroups</th>
      <th>maxPlaceNorm</th>
      <th>kills</th>
      <th>assists</th>
      <th>playersJoined</th>
      <th>maxPlace</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>565181</th>
      <td>1086.00</td>
      <td>29</td>
      <td>1277.538</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>187.4</td>
      <td>1446.64</td>
      <td>1352</td>
      <td>1.07</td>
      <td>10.590</td>
      <td>0.000</td>
      <td>46</td>
      <td>50.29</td>
      <td>1</td>
      <td>0</td>
      <td>93</td>
      <td>47</td>
    </tr>
    <tr>
      <th>1278768</th>
      <td>42.73</td>
      <td>70</td>
      <td>42.730</td>
      <td>0</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1472.04</td>
      <td>1363</td>
      <td>0.00</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>86</td>
      <td>99.36</td>
      <td>0</td>
      <td>0</td>
      <td>92</td>
      <td>92</td>
    </tr>
    <tr>
      <th>884983</th>
      <td>2972.00</td>
      <td>23</td>
      <td>2972.000</td>
      <td>4</td>
      <td>4</td>
      <td>2</td>
      <td>6</td>
      <td>170.1</td>
      <td>0.0</td>
      <td>1325.00</td>
      <td>1250</td>
      <td>1.06</td>
      <td>29.140</td>
      <td>180.306</td>
      <td>27</td>
      <td>28.62</td>
      <td>1</td>
      <td>1</td>
      <td>94</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3040251</th>
      <td>2661.00</td>
      <td>22</td>
      <td>2661.000</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>7</td>
      <td>273.2</td>
      <td>0.0</td>
      <td>1402.50</td>
      <td>1375</td>
      <td>1.02</td>
      <td>37.580</td>
      <td>278.664</td>
      <td>27</td>
      <td>28.56</td>
      <td>1</td>
      <td>0</td>
      <td>98</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2285385</th>
      <td>32.64</td>
      <td>44</td>
      <td>32.640</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>178.0</td>
      <td>0.0</td>
      <td>1568.32</td>
      <td>1508</td>
      <td>1.04</td>
      <td>9.181</td>
      <td>185.120</td>
      <td>48</td>
      <td>50.96</td>
      <td>1</td>
      <td>1</td>
      <td>96</td>
      <td>49</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 重新制定训练集和测试集
df_keep = df[to_keep]
X_train, X_valid = split_vals(df_keep, n_trn)
```


```python
# 模型训练
m2 = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features='sqrt',
                          n_jobs=-1)
# n_jobs=-1 表示训练的时候，并行数和cpu的核数一样，如果传入具体的值，表示用几个核去跑

m2.fit(X_train, y_train)
```




    RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='sqrt', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=3, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=-1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)




```python
# 模型评分
y_pre = m2.predict(X_valid)
m2.score(X_valid, y_valid)
```




    0.9264190806368511




```python
# mae评估
mean_absolute_error(y_true=y_valid, y_pred=y_pre)
```




    0.059019711933656814




```python
print(m2.score)
```

    <bound method RegressorMixin.score of RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
               max_features='sqrt', max_leaf_nodes=None,
               min_impurity_decrease=0.0, min_impurity_split=None,
               min_samples_leaf=3, min_samples_split=2,
               min_weight_fraction_leaf=0.0, n_estimators=40, n_jobs=-1,
               oob_score=False, random_state=None, verbose=0, warm_start=False)>



```python

```


```python

```
