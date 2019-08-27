### 配置

```python
import os
# 配置spark driver和pyspark运行时，所使用的python解释器路径
PYSPARK_PYTHON = "/miniconda2/envs/py365/bin/python"
JAVA_HOME='/root/bigdata/jdk'
SPARK_HOME = "/root/bigdata/spark"
# 当存在多个版本时，不指定很可能会导致出错
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON
os.environ['JAVA_HOME']=JAVA_HOME
os.environ["SPARK_HOME"] = SPARK_HOME
# spark配置信息
from pyspark import SparkConf
from pyspark.sql import SparkSession

SPARK_APP_NAME = "preprocessingBehaviorLog"
SPARK_URL = "spark://192.168.19.137:7077"

conf = SparkConf()    # 创建spark config对象
config = (
    ("spark.app.name", SPARK_APP_NAME),    # 设置启动的spark的app名称，没有提供，将随机产生一个名称
    ("spark.executor.memory", "2g"),    # 设置该app启动时占用的内存用量，默认1g
    ("spark.master", SPARK_URL),    # spark master的地址
    ("spark.executor.cores", "2"),    # 设置spark executor使用的CPU核心数
)
# 查看更详细配置及说明：https://spark.apache.org/docs/latest/configuration.html

conf.setAll(config)

# 利用config对象，创建spark session
spark = SparkSession.builder.config(conf=conf).getOrCreate()
```


```python
spark
```





            <div>
                <p><b>SparkSession - in-memory</b></p>
                
        <div>
            <p><b>SparkContext</b></p>
    
            <p><a href="http://192.168.19.137:4040">Spark UI</a></p>
    
            <dl>
              <dt>Version</dt>
                <dd><code>v2.2.2</code></dd>
              <dt>Master</dt>
                <dd><code>spark://192.168.19.137:7077</code></dd>
              <dt>AppName</dt>
                <dd><code>preprocessingBehaviorLog</code></dd>
            </dl>
        </div>
        
            </div>




### 切割数据


```python
import pandas as pd
reader = pd.read_csv('/root/tmp/behavior_log.csv', chunksize=100,iterator=True)
counter = 0
for chunk in reader:
    counter += 1
    if counter ==1:
        chunk.to_csv('/root/tmp/test_behavior.csv',index = False)
    elif counter>1 and counter<=100000:
        chunk.to_csv('/root/tmp/test_behavior.csv',index = False,header = False,mode = 'a')
    else:
        break
```


```python
# 加载数据用户行为数据
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType
# 构建结构对象
schema = StructType([
    StructField("userId", IntegerType()),
    StructField("timestamp", LongType()),
    StructField("btag", StringType()),
    StructField("cateId", IntegerType()),
    StructField("brandId", IntegerType())
])
# 从hdfs加载数据为dataframe，并设置结构
behavior_log_df = spark.read.csv("/tmp/test_behavior.csv", header=True, schema=schema)
behavior_log_df.show()
```

    +------+----------+----+------+-------+
    |userId| timestamp|btag|cateId|brandId|
    +------+----------+----+------+-------+
    |558157|1493741625|  pv|  6250|  91286|
    |558157|1493741626|  pv|  6250|  91286|
    |558157|1493741627|  pv|  6250|  91286|
    |728690|1493776998|  pv| 11800|  62353|
    |332634|1493809895|  pv|  1101| 365477|
    |857237|1493816945|  pv|  1043| 110616|
    |619381|1493774638|  pv|   385| 428950|
    |467042|1493772641|  pv|  8237| 301299|
    |467042|1493772644|  pv|  8237| 301299|
    |991528|1493780710|  pv|  7270| 274795|
    |991528|1493780712|  pv|  7270| 274795|
    |991528|1493780712|  pv|  7270| 274795|
    |991528|1493780712|  pv|  7270| 274795|
    |991528|1493780714|  pv|  7270| 274795|
    |991528|1493780765|  pv|  7270| 274795|
    |991528|1493780714|  pv|  7270| 274795|
    |991528|1493780765|  pv|  7270| 274795|
    |991528|1493780764|  pv|  7270| 274795|
    |991528|1493780633|  pv|  7270| 274795|
    |991528|1493780764|  pv|  7270| 274795|
    +------+----------+----+------+-------+
    only showing top 20 rows




```python
# 查看数据条数
behavior_log_df.count()
# 查看每个用户的数据条数
behavior_log_df.groupBy('userId').count().show()
# 查看总用户数量
behavior_log_df.groupBy('userId').count().count()
# 查看每种行为类别数量
behavior_log_df.groupBy('btag').count().show()
# 查看类别数量
behavior_log_df.groupBy('cateId').count().count()
# 查看品牌数量
behavior_log_df.groupBy('brandId').count().count()
# 查看删除缺失的数量条数, 看看数据是否有缺失值
behavior_log_df.dropna().count()


```

    +-------+-----+
    | userId|count|
    +-------+-----+
    |  57178|   54|
    | 747610|    8|
    |1081474|    3|
    | 694901|   10|
    | 920750|   70|
    |  35361|   16|
    | 961035|    5|
    | 155430|    6|
    | 493421|   49|
    | 897491|    4|
    |  47217|   18|
    | 976417|   28|
    | 295048|    7|
    | 876272|   26|
    | 495753|    8|
    | 691528|   52|
    |1131890|   98|
    | 763546|    1|
    | 753966|    6|
    | 300779|    9|
    +-------+-----+
    only showing top 20 rows
    
    +----+-------+
    |btag|  count|
    +----+-------+
    | buy| 130132|
    | fav| 129493|
    |cart| 220382|
    |  pv|9519993|
    +----+-------+
    
    ······





### 统计 每一个用户 对不同类别的用户行为数量 


```python
# 统计 每一个用户 对不同类别的用户行为数量 
# user-cate rating   user-brand rating
cate_count_df = behavior_log_df.groupBy(behavior_log_df.userId,behavior_log_df.cateId).pivot('btag').count()

cate_count_df.show()
```

    +-------+------+----+----+----+----+
    | userId|cateId| buy|cart| fav|  pv|
    +-------+------+----+----+----+----+
    | 738396|  9687|null|   1|null|  13|
    | 590694|  6432|null|null|null|   3|
    |1056983|  4283|null|   1|null|  31|
    | 950232|  5953|null|null|null|   1|
    | 386718|  9295|null|null|null|  14|
    | 420528|  9494|null|null|null|  25|
    | 556217|  4520|null|null|null|   6|
    | 985125|  8882|   1|null|null|null|
    | 199025|  6251|null|   2|null|   2|
    |  99730|  4267|null|null|   1|   2|
    | 244032|  6421|null|null|null|   1|
    | 144924|   591|null|null|null|   4|
    | 970608|  5139|null|null|null|   4|
    |  95196|  4505|null|null|null|   1|
    | 275462|  6250|null|null|null|   1|
    |1017870|  5751|null|null|   1|   4|
    | 645845|  7146|null|null|null|   1|
    |  42724|  6935|   1|null|null|   5|
    | 256963|  1102|   1|null|null|   1|
    | 982086|  4280|null|   1|null|   9|
    +-------+------+----+----+----+----+
    only showing top 20 rows



### 把用户行为转换为评分


```python
# 设置CheckpointDir
spark.sparkContext.setCheckpointDir("/checkPoint/")
# 定义行为转换为评分函数
def process_row(r):
	#先初始化数据 把null的情况改成0
	pv_count = r.pv if r.pv else 0.0
	fav_count = r.fav if r.fav else 0.0
	cart_count = r.cart if r.cart else 0.0
	buy_count = r.buy if r.buy else 0.0
	
	pv_score = 0.2*pv_count if pv_count<=20 else 4.0
	fav_score = 0.4*fav_count if fav_count<=20 else 8.0
	cart_score = 0.6*cart_count if cart_count<=20 else 12.0
	buy_score = 1.0*buy_count if buy_count<=20 else 20.0
	
	rating = pv_score+fav_score+cart_score+buy_score
	return r.userId,r.cateId,rating
# 生成用户类别评分矩阵
cate_rating_df = cate_count_df.rdd.map(process_row).toDF(['userId','cateId','rating'])
# 查看数据
cate_rating_df.show()
```

    +-------+------+------------------+
    | userId|cateId|            rating|
    +-------+------+------------------+
    | 738396|  9687|               3.2|
    | 590694|  6432|0.6000000000000001|
    |1056983|  4283|               4.6|
    | 950232|  5953|               0.2|
    | 386718|  9295|2.8000000000000003|
    | 420528|  9494|               4.0|
    | 556217|  4520|1.2000000000000002|
    | 985125|  8882|               1.0|
    | 199025|  6251|               1.6|
    |  99730|  4267|               0.8|
    | 244032|  6421|               0.2|
    | 144924|   591|               0.8|
    | 970608|  5139|               0.8|
    |  95196|  4505|               0.2|
    | 275462|  6250|               0.2|
    |1017870|  5751|1.2000000000000002|
    | 645845|  7146|               0.2|
    |  42724|  6935|               2.0|
    | 256963|  1102|               1.2|
    | 982086|  4280|               2.4|
    +-------+------+------------------+
    only showing top 20 rows




```python
from pyspark.ml.recommendation import ALS
# 创建ALS对象，需要传入DataFrame 指定 用户id是哪一列, 物品的ID是哪一列，评分是哪一列
als = ALS(userCol = 'userId',itemCol = 'cateId',ratingCol = 'rating',checkpointInterval=10)
model = als.fit(cate_rating_df)
```


```python
#保存模型
model.save('/models/test.obj')
```


```python
#模型的保存和加载
from pyspark.ml.recommendation import ALSModel
als_model = ALSModel.load('/models/test.obj')
```


```python
# 为每一个用户推荐三个物品 这里是3个类别
result = model.recommendForAllUsers(3)
```


```python
# 展示数据: 不显示省略号
result.show(truncate = False)
```

    +------+---------------------------------------------------------+
    |userId|recommendations                                          |
    +------+---------------------------------------------------------+
    |463   |[[9442,3.0464783], [7494,3.0187404], [12000,2.781694]]   |
    |471   |[[204,1.5281518], [1796,1.3388014], [9442,1.2715584]]    |
    |496   |[[4783,0.2546865], [2502,0.20388854], [2945,0.20319968]] |
    |833   |[[8666,3.2045805], [4525,2.65586], [12000,2.5573173]]    |
    |1088  |[[6133,0.5033029], [11759,0.45456216], [1796,0.4443066]] |
    |1238  |[[9442,3.388245], [2614,3.2262526], [4667,2.9837246]]    |
    |1342  |[[1566,3.3633032], [650,3.0263388], [1791,2.8940187]]    |
    |1580  |[[8666,0.3696822], [9442,0.32394454], [2614,0.2957789]]  |
    |1591  |[[9915,0.38759547], [3015,0.32911357], [1713,0.32586873]]|
    |1645  |[[9442,0.5249568], [7777,0.48252714], [6375,0.46478167]] |
    |1829  |[[7777,1.9800208], [1796,1.93807], [2614,1.7408638]]     |
    |1959  |[[2614,0.6845258], [7779,0.6334036], [7114,0.6059738]]   |
    |2142  |[[1566,1.0239938], [4470,0.95601416], [1856,0.94817734]] |
    |2659  |[[10149,6.630206], [3612,6.382075], [9442,6.070031]]     |
    |3794  |[[5731,0.47693115], [7114,0.4740107], [2614,0.45692474]] |
    |3918  |[[9442,0.31945923], [4667,0.2581509], [6133,0.23962323]] |
    |3997  |[[9805,0.5324134], [8412,0.4344117], [2502,0.43198377]]  |
    |4519  |[[1460,4.8282146], [7114,4.1858764], [2356,4.111002]]    |
    |4900  |[[7777,2.287503], [6375,2.00171], [4472,1.9902467]]      |
    |4935  |[[204,1.6057427], [7453,1.5214827], [2096,1.3768282]]    |
    +------+---------------------------------------------------------+
    only showing top 20 rows




```python

```


```python

```


```python

```


```python

```


```python

```
