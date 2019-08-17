

```python
from surprise import KNNBasic,SVD
from surprise import Dataset
from surprise import evaluate, print_perf
# http://surprise.readthedocs.io/en/stable/index.html
# http://files.grouplens.org/datasets/movielens/ml-100k-README.txt

# Load the movielens-100k dataset (download it if needed),
# and split it into 3 folds for cross-validation.
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

# We'll use the famous KNNBasic algorithm.
algo = KNNBasic()

# Evaluate performances of our algorithm on the dataset.
perf = evaluate(algo, data, measures=['RMSE', 'MAE'])

print_perf(perf)
```

    Evaluating RMSE, MAE of algorithm KNNBasic.
    
    ------------
    Fold 1
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    RMSE: 0.9876
    MAE:  0.7807
    ------------
    Fold 2
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    RMSE: 0.9871
    MAE:  0.7796
    ------------
    Fold 3
    Computing the msd similarity matrix...
    Done computing similarity matrix.
    RMSE: 0.9902
    MAE:  0.7818
    ------------
    ------------
    Mean RMSE: 0.9883
    Mean MAE : 0.7807
    ------------
    ------------
            Fold 1  Fold 2  Fold 3  Mean    
    MAE     0.7807  0.7796  0.7818  0.7807  
    RMSE    0.9876  0.9871  0.9902  0.9883  
    


```python
from surprise import GridSearch

param_grid = {'n_epochs': [5, 10], 'lr_all': [0.002, 0.005],
              'reg_all': [0.4, 0.6]}
grid_search = GridSearch(SVD, param_grid, measures=['RMSE', 'FCP'])
data = Dataset.load_builtin('ml-100k')
data.split(n_folds=3)

grid_search.evaluate(data)
```

    ------------
    Parameters combination 1 of 8
    params:  {'lr_all': 0.002, 'n_epochs': 5, 'reg_all': 0.4}
    ------------
    Mean RMSE: 0.9972
    Mean FCP : 0.6843
    ------------
    ------------
    Parameters combination 2 of 8
    params:  {'lr_all': 0.005, 'n_epochs': 5, 'reg_all': 0.4}
    ------------
    Mean RMSE: 0.9734
    Mean FCP : 0.6946
    ------------
    ------------
    Parameters combination 3 of 8
    params:  {'lr_all': 0.002, 'n_epochs': 10, 'reg_all': 0.4}
    ------------
    Mean RMSE: 0.9777
    Mean FCP : 0.6926
    ------------
    ------------
    Parameters combination 4 of 8
    params:  {'lr_all': 0.005, 'n_epochs': 10, 'reg_all': 0.4}
    ------------
    Mean RMSE: 0.9635
    Mean FCP : 0.6987
    ------------
    ------------
    Parameters combination 5 of 8
    params:  {'lr_all': 0.002, 'n_epochs': 5, 'reg_all': 0.6}
    ------------
    Mean RMSE: 1.0029
    Mean FCP : 0.6875
    ------------
    ------------
    Parameters combination 6 of 8
    params:  {'lr_all': 0.005, 'n_epochs': 5, 'reg_all': 0.6}
    ------------
    Mean RMSE: 0.9820
    Mean FCP : 0.6953
    ------------
    ------------
    Parameters combination 7 of 8
    params:  {'lr_all': 0.002, 'n_epochs': 10, 'reg_all': 0.6}
    ------------
    Mean RMSE: 0.9860
    Mean FCP : 0.6943
    ------------
    ------------
    Parameters combination 8 of 8
    params:  {'lr_all': 0.005, 'n_epochs': 10, 'reg_all': 0.6}
    ------------
    Mean RMSE: 0.9733
    Mean FCP : 0.6991
    ------------
    


```python
# best RMSE score
print(grid_search.best_score['RMSE'])

# combination of parameters that gave the best RMSE score
print(grid_search.best_params['RMSE'])


# best FCP score
print(grid_search.best_score['FCP'])


# combination of parameters that gave the best FCP score
print(grid_search.best_params['FCP'])

```

    0.963501988854
    {'lr_all': 0.005, 'n_epochs': 10, 'reg_all': 0.4}
    0.699084153002
    {'lr_all': 0.005, 'n_epochs': 10, 'reg_all': 0.6}
    


```python
import pandas as pd  

results_df = pd.DataFrame.from_dict(grid_search.cv_results)
results_df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>FCP</th>
      <th>RMSE</th>
      <th>lr_all</th>
      <th>n_epochs</th>
      <th>params</th>
      <th>scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.684266</td>
      <td>0.997160</td>
      <td>0.002</td>
      <td>5</td>
      <td>{'lr_all': 0.002, 'n_epochs': 5, 'reg_all': 0.4}</td>
      <td>{'RMSE': 0.997160189649, 'FCP': 0.684266412476}</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.694552</td>
      <td>0.973383</td>
      <td>0.005</td>
      <td>5</td>
      <td>{'lr_all': 0.005, 'n_epochs': 5, 'reg_all': 0.4}</td>
      <td>{'RMSE': 0.973383132387, 'FCP': 0.694551932996}</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.692616</td>
      <td>0.977697</td>
      <td>0.002</td>
      <td>10</td>
      <td>{'lr_all': 0.002, 'n_epochs': 10, 'reg_all': 0.4}</td>
      <td>{'RMSE': 0.977696629511, 'FCP': 0.692615513155}</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.698722</td>
      <td>0.963502</td>
      <td>0.005</td>
      <td>10</td>
      <td>{'lr_all': 0.005, 'n_epochs': 10, 'reg_all': 0.4}</td>
      <td>{'RMSE': 0.963501988854, 'FCP': 0.698721750945}</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.687482</td>
      <td>1.002855</td>
      <td>0.002</td>
      <td>5</td>
      <td>{'lr_all': 0.002, 'n_epochs': 5, 'reg_all': 0.6}</td>
      <td>{'RMSE': 1.00285516237, 'FCP': 0.687481665759}</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.695337</td>
      <td>0.982047</td>
      <td>0.005</td>
      <td>5</td>
      <td>{'lr_all': 0.005, 'n_epochs': 5, 'reg_all': 0.6}</td>
      <td>{'RMSE': 0.98204676013, 'FCP': 0.695337489535}</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.694338</td>
      <td>0.985981</td>
      <td>0.002</td>
      <td>10</td>
      <td>{'lr_all': 0.002, 'n_epochs': 10, 'reg_all': 0.6}</td>
      <td>{'RMSE': 0.985980855401, 'FCP': 0.694337564062}</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.699084</td>
      <td>0.973282</td>
      <td>0.005</td>
      <td>10</td>
      <td>{'lr_all': 0.005, 'n_epochs': 10, 'reg_all': 0.6}</td>
      <td>{'RMSE': 0.973281870802, 'FCP': 0.699084153002}</td>
    </tr>
  </tbody>
</table>
</div>




```python
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import os
import io

from surprise import KNNBaseline
from surprise import Dataset


def read_item_names():


    file_name = ('./ml-100k/u.item')
    rid_to_name = {}
    name_to_rid = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
            name_to_rid[line[1]] = line[0]

    return rid_to_name, name_to_rid



data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
sim_options = {'name': 'pearson_baseline', 'user_based': False}
algo = KNNBaseline(sim_options=sim_options)
algo.train(trainset)
```

    Estimating biases using als...
    Computing the pearson_baseline similarity matrix...
    Done computing similarity matrix.
    


```python
rid_to_name, name_to_rid = read_item_names()

toy_story_raw_id = name_to_rid['Now and Then (1995)']
toy_story_raw_id
```




    '1053'




```python
toy_story_inner_id = algo.trainset.to_inner_iid(toy_story_raw_id)
toy_story_inner_id
```




    961




```python
toy_story_neighbors = algo.get_neighbors(toy_story_inner_id, k=10)
toy_story_neighbors
```




    [291, 82, 366, 528, 179, 101, 556, 310, 431, 543]




```python
toy_story_neighbors = (algo.trainset.to_raw_iid(inner_id)
                       for inner_id in toy_story_neighbors)
toy_story_neighbors = (rid_to_name[rid]
                       for rid in toy_story_neighbors)

print()
print('The 10 nearest neighbors of Toy Story are:')
for movie in toy_story_neighbors:
    print(movie)
```

    
    The 10 nearest neighbors of Toy Story are:
    While You Were Sleeping (1995)
    Batman (1989)
    Dave (1993)
    Mrs. Doubtfire (1993)
    Groundhog Day (1993)
    Raiders of the Lost Ark (1981)
    Maverick (1994)
    French Kiss (1995)
    Stand by Me (1986)
    Net, The (1995)
    


```python

```
