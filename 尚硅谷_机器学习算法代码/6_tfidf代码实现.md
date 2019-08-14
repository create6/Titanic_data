
# TF-IDF算法示例

### 0. 引入依赖


```python
import numpy as np
import pandas as pd
```

### 1. 定义数据和预处理


```python
docA = "The cat sat on my bed"
docB = "The dog sat on my knees"

bowA = docA.split(" ")
bowB = docB.split(" ")
bowA

# 构建词库
wordSet = set(bowA).union(set(bowB))
wordSet
```




    {'The', 'bed', 'cat', 'dog', 'knees', 'my', 'on', 'sat'}



### 2. 进行词数统计


```python
# 用统计字典来保存词出现的次数
wordDictA = dict.fromkeys( wordSet, 0 )
wordDictB = dict.fromkeys( wordSet, 0 )

# 遍历文档，统计词数
for word in bowA:
    wordDictA[word] += 1
for word in bowB:
    wordDictB[word] += 1
    
pd.DataFrame([wordDictA, wordDictB])
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
      <th>The</th>
      <th>bed</th>
      <th>cat</th>
      <th>dog</th>
      <th>knees</th>
      <th>my</th>
      <th>on</th>
      <th>sat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



### 3. 计算词频TF


```python
def computeTF( wordDict, bow ):
    # 用一个字典对象记录tf，把所有的词对应在bow文档里的tf都算出来
    tfDict = {}
    nbowCount = len(bow)
    
    for word, count in wordDict.items():
        tfDict[word] = count / nbowCount
    return tfDict

tfA = computeTF( wordDictA, bowA )
tfB = computeTF( wordDictB, bowB )
tfA
```




    {'cat': 0.16666666666666666,
     'on': 0.16666666666666666,
     'sat': 0.16666666666666666,
     'knees': 0.0,
     'bed': 0.16666666666666666,
     'The': 0.16666666666666666,
     'my': 0.16666666666666666,
     'dog': 0.0}



### 4. 计算逆文档频率idf


```python
def computeIDF( wordDictList ):
    # 用一个字典对象保存idf结果，每个词作为key，初始值为0
    idfDict = dict.fromkeys(wordDictList[0], 0)
    N = len(wordDictList)
    import math
    
    for wordDict in wordDictList:
        # 遍历字典中的每个词汇，统计Ni
        for word, count in wordDict.items():
            if count > 0:
                # 先把Ni增加1，存入到idfDict
                idfDict[word] += 1
                
    # 已经得到所有词汇i对应的Ni，现在根据公式把它替换成为idf值
    for word, ni in idfDict.items():
        idfDict[word] = math.log10( (N+1)/(ni+1) )
    
    return idfDict

idfs = computeIDF( [wordDictA, wordDictB] )
idfs
```




    {'cat': 0.17609125905568124,
     'on': 0.0,
     'sat': 0.0,
     'knees': 0.17609125905568124,
     'bed': 0.17609125905568124,
     'The': 0.0,
     'my': 0.0,
     'dog': 0.17609125905568124}



### 5. 计算TF-IDF


```python
def computeTFIDF( tf, idfs ):
    tfidf = {}
    for word, tfval in tf.items():
        tfidf[word] = tfval * idfs[word]
    return tfidf

tfidfA = computeTFIDF( tfA, idfs )
tfidfB = computeTFIDF( tfB, idfs )

pd.DataFrame( [tfidfA, tfidfB] )
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
      <th>The</th>
      <th>bed</th>
      <th>cat</th>
      <th>dog</th>
      <th>knees</th>
      <th>my</th>
      <th>on</th>
      <th>sat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.029349</td>
      <td>0.029349</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.029349</td>
      <td>0.029349</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
