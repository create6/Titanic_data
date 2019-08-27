```
大数据争夺架构霸权之战中，云计算显然是赢家，为什么学 Hadoop ？Flink与Spark有什么区别？后Hadoop世界中的大数据如何走下去？

云计算底层可能就是通过Hadoop实现的. 
* Flink: 实时流式处理框架, 延迟ms级, 来一条处理一条
* Spark: 
	准时实时流式处理框架, 延迟s级, 每隔1s处理一次.
	spark的社区活跃度比flink强很多，生态更加丰富		    
```

|           | SPARK           | **Flink**                    |
| --------- | --------------- | ---------------------------- |
| 类SQL查询 | Spark SQL       | MRQL                         |
| 图计算    | Spark GraphX    | Spargel（基础）和Gelly（库） |
| 机器学习  | Spark  ML/MLib  | Flink ML                     |
| 流计算    | Spark Streaming | Flink Streaming              |

```
hdfs:  cd bigdata/hadoop/sbin ./start-dfs.sh 关闭安全安全模式: hdfs dfsadmin -safemode leave
mapreduce: cd bigdata/hadoop/sbin ./start-yarn.sh 
hive: 启动hdfs, yarn, mysql(启动docker, 启动mysql) hive
hbase: 启动hdfs, 启动hbase cd bigdata/hbase/bin/  ./start-hbase   进入命令环境 hbase shell 
spark: 启动hdfs, pyspark(本地); 
独立集群: 
	cd bigdata/spark/sbin
	./start-master.sh -h 192.168.19.137
	./start-slave.sh spark://192.168.19.137:7077
```



数据集的下载地址

```
https://tianchi.aliyun.com/dataset/dataDetail?dataId=56
```



### 01_思路介绍[8:17]

- 召回
	- 利用用户行为数据behavior_log创建 协同过滤模型
	- 矩阵分解，Spark提供 ALS
	- 如何把用户的行为转换成评分
	- 召回用户最感兴趣的类别   最感兴趣的品牌
- 排序
	- LR模型   CTR预估 预测的是用户对物品的点击概率
	- raw_sample  用户id 物品id 时间戳 广告资源位
		- **点/没点**
	- ad_feature      物品的特征
	- user_profile     用户特征
- 缓存数据
	- 用户特征
	- 用户感兴趣的物品 用户感兴趣的类别=》类别下的物品   用户感兴趣的品牌=》品牌下的物品 
	- 每一个物品的特征
- 线上推荐
  - 加载缓存数据 带入到排序模型 实时排序



### 数据处理

- 了解基本数据的情况
  - 数据量
  - 有哪些字段 类型 分类/连续   
  - 有没有缺失
- one-hot
- 缺失值处理
  - 利用机器学习的算法 预测缺失值
  - 如果是分类特征，保留缺失值做为单独的分类
- Spark ML /  Spark MLlib

### 02_数据拆分[11:03]
- 读取数据
	-  pd.read_csv(path_or_buf, chunksize=100,iterator=True)
		- 作用: 读取csv数据
		- 参数:
			- path_or_buf: 资源路径
			- chunksize: 每次读多少条数据
			- iterator: 是否迭代获取数据
	- DataFrame.to_csv(path_or_buf=None, index=True,header=True,  mode="w")
		- 作用: 以csv格式写入数据
		- 参数:
			- path_or_buf: 资源路径
			- index: 是否带行索引, 默认是True
			- header: 是否带列索引, 默认True
			- mode: 写模式, 默认`w` 覆盖写, `a`:追加写
- 代码
  ```python
	import pandas as pd
	reader = pd.read_csv('/root/tmp/data/behavior_log.csv', chunksize=100,iterator=True)
	counter = 0
	for chunk in reader:
		counter += 1
		if counter ==1:
			chunk.to_csv('/root/tmp/data/test.csv',index = False)
		elif counter>1 and counter<=100000:
			chunk.to_csv('/root/tmp/data/test.csv',index = False,header = False,mode = 'a')
		else:
			break
  ```

- 附: 建立Master的链接
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



### 03_用户行为数据情况调查[18:27]

- 目标: 查看用户行为情况
- 步骤:
	- 加载数据
	- 查看数据总条数
	- 查看用户数量
	- 查看类别数量
	- 查看品牌数量
	- 查看是否有缺失值
	- 统计 每一个用户 对不同行为的数量 
- 代码
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
behavior_log_df = spark.read.csv("/data/test1.csv", header=True, schema=schema)
behavior_log_df.show()

# 查看数据条数
df.count()
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

# 统计 每一个用户 对不同类别的用户行为数量 
# user-cate rating   user-brand rating
cate_count_df = behavior_log_df.groupBy(behavior_log_df.userId,behavior_log_df.cateId).pivot('btag').count()

cate_count_df.show()
```



### 04_ALS模型创建数据准备[13:30]

- 目标: 把用户行为转换为评分
- 步骤:
	- 设置检查点目录
	- 定义行为转换为评分函数
	- 生成用户类别评分数据
- 代码:
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



### 05_ALS模型创建[18:27]

- 目标: 能够创建,保存,加载ALS模型
- SparkML /Spark MLlib
  - SparkML 更新 基于DataFrame, 目前如果还对机器学习的内容进行跟新，只会更新SparkML
  - Spark MLlib 基于RDD的 ，处于维护状态，不会更新了
- 步骤:
	1. 加载模型
	2. 使用模型进行TopN推荐
	3. 把推荐结果保存Redis中
- 创建和训练ALS模型
  ```python
  from pyspark.ml.recommendation import ALS
	# 创建ALS对象，需要传入DataFrame 指定 用户id是哪一列, 物品的ID是哪一列，评分是哪一列
	als = ALS(userCol = 'userId',itemCol = 'cateId',ratingCol = 'rating',checkpointInterval=10)
	model = als.fit(cate_rating_df)
  ```
- 保存模型
  ```python
	model.save('/models/test.obj')
  ```
- 加载模型
  ```python
	#模型的保存和加载
	from pyspark.ml.recommendation import ALSModel
	als_model = ALSModel.load('/models/userCateRatingALSModel.obj')
  ```
- 为每一个用户推荐商品
  	```python
	# 为每一个用户推荐三个物品 这里是3个类别
	result = model.recommendForAllUsers(3)
	```
- 显示推荐的数据
	```python
	# 展示数据: 不显示省略号
	result.show(truncate = False)
	```
- 把数据保存Redis中
  ```python
	import redis
	host = '192.168.2.137'
	port = 6379
	def recall_cate_by_cf(partition):
		#建立redis连接池
		pool = redis.ConnectionPool(host = host,port=port)
		#建立redis客户端
		client = redis.Redis(connection_pool=pool)
		#遍历partition中的数据
		for row in partition:
			client.hset('recall_cate1',row.userId,[i.cateId for i in row.recommendations])
	
	    
	result.foreachPartition(recall_cate_by_cf)
	```

* 启动Redis服务
  * redis-server /etc/redis.conf 



### 06_CTR模型创建数据准备_onehot处理

- 目标: 能够对数据进行onehot处理
- 知识点
	- StringIndexer
		- 把分类数据，转换成从0开始的分类值
		- 指定输入的列，和输出列名
		- StringIndexer.fit(dataframe)
		- StringIndexer.transform(dataframe)
	- OneHotEncoder
		- 在StringIndexer基础上，把数据处理成OneHot的形式
		- 输出的OneHot的结果，使用SparseVector（稀疏向量）的形式来表示的
		- 创建OneHotEncoder对象的时候 有一个参数droplast 一般传入False，如果传入True
			那么 会丢掉OneHot的最后一位，也就是说，三分类的数据使用2维向量来表示
			[1, 0, 0], [0,1,0],[0,0,1] =>  [1,0] ,[0,1],[0,0]
		- OneHotEncoder.transform(dataframe)
	- Pipeline
		- 让数据按顺序依次被处理，将前一次的处理结果作为下一次的输入
		- 可以定义多个stage 减少fit transform的次数

- 步骤
	- 加载数据
	- 修改schema
	- 对数据进行OneHot编码
- 代码
	```python
	# 加载数据
	df = spark.read.csv('/data/raw_sample.csv',header = True)
	# 更改表结构，转换为对应的数据类型
	from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, LongType, StringType

	# 打印df结构信息
	df.printSchema()   
	# 更改df表结构：更改列类型和列名称
	raw_sample_df = df.\
		withColumn("user", df.user.cast(IntegerType())).withColumnRenamed("user", "userId").\
		withColumn("time_stamp", df.time_stamp.cast(LongType())).withColumnRenamed("time_stamp", "timestamp").\
		withColumn("adgroup_id", df.adgroup_id.cast(IntegerType())).withColumnRenamed("adgroup_id", "adgroupId").\
		withColumn("pid", df.pid.cast(StringType())).\
		withColumn("nonclk", df.nonclk.cast(IntegerType())).\
		withColumn("clk", df.clk.cast(IntegerType()))
	raw_sample_df.printSchema()
	raw_sample_df.show()

	# StringIndexer对指定字符串列进行特征处理
	stringindexer = StringIndexer(inputCol='pid', outputCol='pid_feature')
	# 对处理出来的特征处理列进行，热独编码
	encoder = OneHotEncoder(dropLast=False, inputCol='pid_feature', outputCol='pid_value')
	# 利用管道对每一个数据进行热独编码处理
	pipeline = Pipeline(stages=[stringindexer, encoder])
	pipeline_model = pipeline.fit(raw_sample_df)
	new_df = pipeline_model.transform(raw_sample_df)
	new_df.show()
	```



### 07_CTR模型创建数据准备_商品特征数据处理[8:44]
- 目标: 处理ad_feature数据
- 步骤:
	- 加载数据
	- 修改schema
	- 查看数据情况
	- 对价格进行排序
	- 统计价格情况
- 代码
	```python
	# 加载数据
	df = spark.read.csv('/data/ad_feature.csv',header = True)
	df.show()

	# 修改schema
	from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

	# 替换掉NULL字符串，替换掉
	df = df.replace("NULL", "-1")
	# 打印df结构信息
	df.printSchema()   
	# 更改df表结构：更改列类型和列名称
	ad_feature_df = df.\
		withColumn("adgroup_id", df.adgroup_id.cast(IntegerType())).withColumnRenamed("adgroup_id", "adgroupId").\
		withColumn("cate_id", df.cate_id.cast(IntegerType())).withColumnRenamed("cate_id", "cateId").\
		withColumn("campaign_id", df.campaign_id.cast(IntegerType())).withColumnRenamed("campaign_id", "campaignId").\
		withColumn("customer", df.customer.cast(IntegerType())).withColumnRenamed("customer", "customerId").\
		withColumn("brand", df.brand.cast(IntegerType())).withColumnRenamed("brand", "brandId").\
		withColumn("price", df.price.cast(FloatType()))
	ad_feature_df.printSchema()
	ad_feature_df.show()
	# 查看数据情况
	print("总广告条数：",df.count())   # 数据条数
	_1 = ad_feature_df.groupBy("cateId").count().count()
	print("cateId数值个数：", _1)
	_2 = ad_feature_df.groupBy("campaignId").count().count()
	print("campaignId数值个数：", _2)
	_3 = ad_feature_df.groupBy("customerId").count().count()
	print("customerId数值个数：", _3)
	_4 = ad_feature_df.groupBy("brandId").count().count()
	print("brandId数值个数：", _4)
	# 数据 价格 排序
	ad_feature_df.sort('price').show()
	ad_feature_df.sort('price',ascending = False).show()
	# 查看价格大于10000商品价格数据
	ad_feature_df.select('price').filter('price>10000').count()
	# 查看价格小于1商品价格数据
	ad_feature_df.select('price').filter('price<1').count()
	```

结论: 广告特征只要价格

### 08_CTR模型创建数据准备_用户特征数据基本情况分析[07:45]

- 目标: 知道如何查看数据情况
- 步骤:
	- 加载数据
	- 查看分类情况
	- 查看缺失值情况
	- 查看缺失值比例
- 代码
	```python
	from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, FloatType

	# 构建表结构schema对象
	schema = StructType([
		StructField("userId", IntegerType()),
		StructField("cms_segid", IntegerType()),
		StructField("cms_group_id", IntegerType()),
		StructField("final_gender_code", IntegerType()),
		StructField("age_level", IntegerType()),
		StructField("pvalue_level", IntegerType()),
		StructField("shopping_level", IntegerType()),
		StructField("occupation", IntegerType()),
		StructField("new_user_class_level", IntegerType())
	])
	# 利用schema从hdfs加载
	user_profile_df = spark.read.csv("/data/user_profile.csv", header=True, schema=schema)
	user_profile_df.printSchema()
	user_profile_df.show()

	print("分类特征值个数情况: ")
	print("cms_segid: ", user_profile_df.groupBy("cms_segid").count().count())
	print("cms_group_id: ", user_profile_df.groupBy("cms_group_id").count().count())
	print("final_gender_code: ", user_profile_df.groupBy("final_gender_code").count().count())
	print("age_level: ", user_profile_df.groupBy("age_level").count().count())
	print("shopping_level: ", user_profile_df.groupBy("shopping_level").count().count())
	print("occupation: ", user_profile_df.groupBy("occupation").count().count())

	print('含有缺失值的特征情况：')
	user_profile_df.groupBy('pvalue_level').count().show()
	user_profile_df.groupBy('new_user_class_level').count().show()

	# 查看缺失值的比例
	# 总数据条数
	t_count = user_profile_df.count()
	# 消费档次 缺失值数量
	pl_na_count = t_count-user_profile_df.dropna(subset=['pvalue_level']).count()
	# 计算消费档次 缺失比例
	pl_na_count/t_count

	# 城市层级 缺失值数量
	nul_na_count = t_count-user_profile_df.dropna(subset=['new_user_class_level']).count()
	# 计算城市层级 缺失比例
	nul_na_count/t_count
	```

### 09_CTR模型创建数据准备_用户特征数据缺失值处理[10:42]
- 目标: 知道处理用户特征缺失值的思路
- 步骤
	- 使用剔除缺失值数据, 将剩余的数据作为训练集, 需要把数据处理成LabeledPoint
	- 训练随机森林模型
	- 获取缺失值数据
	- 使用随机森林模型, 对缺失值进行预测
	- 利用预测的结果回填原始数据
- 知识点

- 数据的准备，LabeledPoint
  - 在MLlib中，LabeledPoint用于监督学习算法

  - 在训练模型之前，需要把数据处理成由LabeledPoint对象组成的RDD

  - 创建LabeledPoint(1.0, [1.0, 0.0, 3.0])
  	```python
  		from pyspark.mllib.linalg import SparseVector
  		from pyspark.mllib.regression import LabeledPoint
  		
  		# Create a labeled point with a positive label and a dense feature vector.
  		#第一个参数，目标值，如果目标值是分类值，必须是从0开始的
  		#第二个参数 特征的list
  		pos = LabeledPoint(1.0, [1.0, 0.0, 3.0])
  		
  		# Create a labeled point with a negative label and a sparse feature vector.
  		neg = LabeledPoint(0.0, SparseVector(3, [0, 2], [1.0, 3.0]))
  	```
  	
  - 训练随机森林模型

    ```python
    pyspark.mllib.tree import RandomForest
    # 训练分类模型
    # 参数1 训练的数据
    #参数2 目标值的分类个数 0,1,2
    #参数3 特征中是否包含分类的特征 {2:2,3:7} {2:2} 表示 在特征中 第二个特征是分类的: 有两个分类
    #参数4 随机森林中 树的棵数
    model = RandomForest.trainClassifier(train_data, 3, {}, 5)
    ```


### 10_CTR模型创建数据准备_用户特征数据缺失值处理2
- 目标: 能够使用随机森林处对缺失值进行预测
- 步骤
	- 使用剔除缺失值数据, 将剩余的数据作为训练集, 需要把数据处理成LabeledPoint
	- 训练随机森林模型
	- 使用随机森林模型, 根据特征数据进行预测

- 代码
	```python
	# 1. 使用剔除缺失值数据, 将剩余的数据作为训练集, 需要把数据处理成LabeledPoint
	# 使用MLlib 做预测，数据需要准备成由Labeled Point对象组成的RDD
	from pyspark.mllib.regression import LabeledPoint

	train_data = user_profile_df.dropna(subset=['pvalue_level']).rdd.map(lambda r:LabeledPoint(r.pvalue_level-1,[r.cms_segid, r.cms_group_id, r.final_gender_code, r.age_level, r.shopping_level, r.occupation]))

	# 2.  训练随机森林模型
	from pyspark.mllib.tree import RandomForest
	model = RandomForest.trainClassifier(train_data,3,{},5)

	# 3. 使用随机森林模型, 对缺失值进行预测
	model.predict([49.0,6.0,2.0,6.0,3.0,0.0])
	```

### 11_CTR模型创建数据准备_用户特征数据缺失值处理完成
- 内容:
	- 总结缺失值处理流程
	- 使用随机森林处理缺失值
- 步骤
	- 使用剔除缺失值数据, 将剩余的数据作为训练集, 需要把数据处理成LabeledPoint
	- 训练随机森林模型
	- 获取缺失值数据
	- 使用随机森林模型, 对缺失值进行预测
	- 利用预测的结果回填原始数据

- 代码

  ```python
	# 获取缺失值数据
	# 取出所有包含缺失值的数据
	pl_na_df = user_profile_df.na.fill(-1).where('pvalue_level=-1')
	# 使用随机森林模型, 对缺失值进行预测
	# 定义函数返回特征值
	def row(r):
		return r.cms_segid, r.cms_group_id, r.final_gender_code, r.age_level, r.shopping_level, r.occupation
	rdd = pl_na_df.rdd.map(row)
	# 对缺失数据进行预测
	predicts = model.predict(rdd)

	# 利用预测的结果回填原始数据
	# 把数据转换为pandas
	temp = predicts.map(lambda x:int(x)).collect()
	 = pl_na_df.toPandas()
	# 把数据添加到缺失数据
	import numpy as np
	pdf['pvalue_level'] = np.array(temp)+1  

	# 利用Pandas的df,  创建spark的df
	tempDF = spark.createDataFrame(pdf,schema=schema)
	# 把缺失的和正常的数据拼接到一起
	new_user_profile_df = user_profile_df.dropna(subset=['pvalue_level']).unionAll(tempDF)
  ```

### 12_CTR模型创建数据准备_缺失值变成一个单独的分类进行Onehot[13:11]
- 目标: 能够把缺失值变成一个单独的分类进行Onehot
- 前提: 特征是分类情况
- 步骤:
	- 把缺失值填充为 -1
	- 把要处理的字段转换为字符串类型
	- 对pvalue_level进行one-hot编码，求值
	- 对new_user_class_level进行one-hot编码
- 代码
  ```python
	# 把缺失值填充为 -1
	from pyspark.sql.types import StringType
	user_profile_df = user_profile_df.na.fill(-1)
	user_profile_df.show()
	# 在进行Onehot处理时 stringindexer要求数据是字符串类型的
	# 把要处理的字段转换为字符串类型
	user_profile_df = user_profile_df.withColumn('pvalue_level',user_profile_df.pvalue_level.cast(StringType())).withColumn('new_user_class_level',user_profile_df.new_user_class_level.cast(StringType()))

	# 对pvalue_level进行热独编码，求值
	stringIndexer1 = StringIndexer(inputCol = 'pvalue_level',outputCol = 'pl_onehot_feature')
	encoder2 = OneHotEncoder(dropLast = False,inputCol = 'pl_onehot_feature',outputCol = 'pl_onehot_value')
	pipeline1 = Pipeline(stages = [stringIndexer1,encoder2])
	temp_model = pipeline1.fit(user_profile_df)
	user_profile_df2 = temp_model.transform(user_profile_df)

	user_profile_df2.show()

	# 对new_user_class_level进行热独编码
	stringindexer = StringIndexer(inputCol='new_user_class_level', outputCol='nucl_onehot_feature')
	encoder = OneHotEncoder(dropLast=False, inputCol='nucl_onehot_feature', outputCol='nucl_onehot_value')
	pipeline = Pipeline(stages=[stringindexer, encoder])
	pipeline_fit = pipeline.fit(user_profile_df2)
	user_profile_df3 = pipeline_fit.transform(user_profile_df2)
	user_profile_df3.show()
  ```
### 13_CTR模型创建数据准备_数据合并[20:23]

- 目标: 能够合并数据并选择出需要的特征
- 步骤
	- 合并使用one-hot编码后的广告点击数据与广告特征数据
	- 合并后的数据再与用户特征数据进行合并
	- 选择出需要的特征
	- 去除缺失值
	- 根据特征字段计算出特征向量，并划分出训练数据集和测试数据集
- 代码:
	```python
		# 合并使用one-hot编码后的广告点击数据与广告特征数据
		condition = [new_df.adgroupId == ad_feature_df.adgroupId]
		_ = new_df.join(ad_feature_df,condition,'outer')
		# 合并后的数据再与用户特征数据进行合并
		condition2 = [_.userId == user_profile_df3.userId]
		datasets = _.join(user_profile_df3,condition2,'outer')
		
		useful_cols = [
			# 
			# 时间字段，划分训练集和测试集
			"timestamp",
			# label目标值字段
			"clk",  
			# 特征值字段
			"pid_value",       # 资源位的特征向量
			"price",    # 广告价格
			"cms_segid",    # 用户微群ID
			"cms_group_id",    # 用户组ID
			"final_gender_code",    # 用户性别特征，[1,2]
			"age_level",    # 年龄等级，1-
			"shopping_level",
			"occupation",
			"pl_onehot_value",
			"nucl_onehot_value"
		]
		# 筛选指定字段数据，构建新的数据集
		datasets_1 = datasets.select(*useful_cols)
		# 由于前面使用的是outer方式合并的数据，产生了部分空值数据，这里必须先剔除掉
		datasets_1 = datasets_1.dropna()

		# 利用数据去训练逻辑回归的模型¶
		# 把数据准备好 创建一个DataFrame 目标值是一列，所有的特征都放到一列中
		# 按时间戳进行排序
		datasets_1.sort('timestamp', ascending = False).show()
	```

### 14_CTR模型训练完成并预测结果[13:06]
- 目标: 能够实现CTR模型训练并预测结果
- 步骤
  - VectorAssemble 把特征值放到一列里面
  - 训练集测试集的划分  利用时间戳这一列
  - 创建逻辑回归分类器对象 训练模型(时间太久, 直接加载一个模型来用)
  - 加载逻辑回归模型

- 代码
	```python
	# 1. VectorAssemble 把特征值放到一列里面
	from pyspark.ml.feature import VectorAssembler
	datasets_1 = VectorAssembler().setInputCols(useful_cols[2:]).setOutputCol('features').transform(datasets_1)

	# 2. 训练集测试集的划分  利用时间戳这一列
	datasets_1.sort('timestamp',ascending = False).show()
	# 训练集
	train_datasets_1 = datasets_1.filter(datasets_1.timestamp<=(1494691186-24*60*60))
	# 测试集
	test_datasets_1 = datasets_1.filter(datasets_1.timestamp>(1494691186-24*60*60))

	#3. 创建逻辑回归分类器对象 训练模型
	from pyspark.ml.classification import LogisticRegression
	lr = LogisticRegression()

	#设置目标值和特征值字段, 并训练模型
	# model = lr.setLabelCol('clk').setFeaturesCol('features').fit(train_datasets_1)
	# 加载逻辑回归模型
	from pyspark.ml.classification import LogisticRegressionModel
	model = LogisticRegressionModel.load("hdfs://192.168.2.137:9000/models/CTRModel_Normal.obj")

	# 根据测试数据进行预测
	result_1 = model.transform(test_datasets_1)
	result_1.show()
	```

### 15_为用户召回商品思路分析[12:11]
- 内容:
  - 逻辑回归思路总结
  - 用户召回商品思路分析

 - 利用SparkML 训练逻辑回归的模型
	- 数据的准备
	- 准备一个DataFrame，这个DataFrame一定要包含两列
		- 一列是目标值
		- 一列是所有的特征的list
	- 可以使用VectorAssembler, 把要使用的特征，从不同列中整理到一列里
		```python
		from pyspark.ml.feature import VectorAssembler
		datasets_1 = VectorAssembler().setInputCols(useful_cols[2:]).setOutputCol('features').transform(datasets_1)
		```
		setInputCols 把所有要用到的特征的列名的list传递进来，setOutputCol 输出的列名
	- 训练逻辑回归模型
		```python
		from pyspark.ml.classification import LogisticRegression
		lr = LogisticRegression()
		#设置目标值和特征值字段
		model = lr.setLabelCol('clk').setFeaturesCol('features').fit(train_datasets_1)
		```
	- 训练好模型了，可以把模型保存起来，也可以使用模型预测结果
		
		- model.transform(test_datasets_1)
- 用户召回商品思路分析
  - 建立类别和广告的对应关系的DataFrame
  - 利用ALS模型进行类别的召回(找到用户感兴趣的类别)
  - 根据感兴趣的类别, 每一个用户中随机选择500个广告作为召回的广告
  - 把召回的广告添加到Redis中



### 16_为用户召回商品完成[17:05]

- 技术点演示

```python
#找到类别和广告的对应关系
# ad_feature_df 广告特征的表
_ = ad_feature_df.select('adgroupId','cateId') 
pdf = _.toPandas()

# 获取指定类别下的广告
pdf.where(pdf.cateId ==11156).dropna().adgroupId # 700个

#根据类别到对应关系表中 随机选择当前类别下的200个广告
np.random.choice(pdf.where(pdf.cateId ==11156).dropna().adgroupId.astype(np.int64),200)

# 给用户找到感兴趣的类别
als_model.userFactors.show()
# 为指定用户预测所有类别的评分
# 生成类别的DF
cateId_df = pd.DataFrame(pdf.cateId.unique(),columns=['cateId'])
# 把指定用户数据插入到类别的df中
cateId_df.insert(0,'userId',np.array([8 for i in range(6769)]))
# 使用ASL模型, 预测用户对类别的评分
als_model.transform(spark.createDataFrame(cateId_df)).sort('prediction',ascending = False).na.drop().show()
```

- 整体思路
	- 遍历UserFactors 找到每一个用户 为每一个用户建立 User 所有类别  dataframe 
	- 利用ALS模型 预测用户对所有类别的评分值 把评分高的排在前面
	- 利用排在前面的类别到 类别-广告的对应关系表中随机抽取商品 抽取出来放到redis当中

- 代码
```python
# 为用户召回商品
# 遍历UserFactors 找到每一个用户 为每一个用户建立 User 所有类别  dataframe 
# 利用ALS模型 预测用户对所有类别的评分值 把评分高的排在前面
# 利用排在前面的类别到 类别-广告的对应关系表中随机抽取商品 抽取出来放到redis当中
import redis
# 建立Redis数据库连接
client = redis.StrictRedis(host = '192.168.19.137',port=6379,db = 6)

for r in als_model.userFactors.select('id').collect():
    # 获取用户ID
	userId = r.id
	# 创建类别ID的DataFrame
    cateId_df = pd.DataFrame(pdf.cateId.unique(),columns=['cateId'])
	# 向类别DataFrame中插入用户ID列
    cateId_df.insert(0,'userId',np.array([userId for i in range(6769)]))

    ret = set()
	# 利用ALS模型, 预测用户对类别评分, 倒叙排列
    cateId_list = als_model.transform(spark.createDataFrame(cateId_df)).sort('prediction',ascending = False).na.drop()
    #从cateId_list取出前20个结果
    for i in cateId_list.head(20):
        need = 500-len(ret)
		# 根据从类别和广告的对应关系中随机选择 need 个广告
        ret = ret.union(np.random.choice(pdf.where(pdf.cateId ==i).dropna().adgroupId.astype(np.int64),need))
        if len(ret)>=500:
            break
	
	# 把用户ID和召回的广告, 添加Redis中
    client.sadd(userId,*ret)
```

### 17_召回商品缓存到redis
- 步骤:
	- 缓存广告的基本信息 
	- 缓存用户的基本信息
- 代码: 

```python

# 缓存商品(广告)的特征 缓存了价格
def foreachPartition(partition):

    import redis
    import json
    client = redis.StrictRedis(host="192.168.2.137", port=6379, db=2)

    for r in partition:
        data = {
            "price": r.price
        }
        # 转成json字符串再保存，能保证数据再次倒出来时，能有效的转换成python类型
        client.hset("ad_features", r.adgroupId, json.dumps(data))

ad_feature_df.foreachPartition(foreachPartition)

# 缓存用户的基本信息
def foreachPartition2(partition):

    import redis
    import json
    client = redis.StrictRedis(host="192.168.2.137", port=6379, db=3)

    for r in partition:
        data = {
            "cms_segid": r.cms_segid,
            "cms_group_id": r.cms_group_id,
            "final_gender_code": r.final_gender_code,
            "age_level": r.age_level,
            "shopping_level": r.shopping_level,
            "occupation": r.occupation,
            "pvalue_level": r.pvalue_level,
            "new_user_class_level": r.new_user_class_level
        }
        # 转成json字符串再保存，能保证数据再次倒出来时，能有效的转换成python类型
        client.hset("user_features1", r.userId, json.dumps(data))
user_profile_df.foreachPartition(foreachPartition2)
```

### 18_加载缓存特征[09:12]
- 目标: 根据用户ID和广告资源位 返回前用户的召回集对应的物品特征，以及用户特特征
- 步骤
	- 准备真实值与one-hot编码后的特征的对应关系对应关系
	- 定义方法, 根据用户ID和广告资源位 返回前用户的召回集对应的物品特征，以及用户特特征
```python

# 真实值 与 特征值的对应关系;
# 对应关系来自于one-hot编码后的内容的观察, 总结出来的.
pvalue_level_rela = {-1:0, 3:3, 1:2, 2:1}
new_user_class_level_rela = {-1:0, 3:2, 1:4, 4:3, 2:1}
pid_rela = {"430548_1007": 0, "430549_1007": 1}


import redis
import json
import pandas as pd
from pyspark.ml.linalg import DenseVector

# 需要传递两个值   
def create_datasets(userId, pid):
	# 加载用户召回广告数据
    client_of_recall = redis.StrictRedis(host="192.168.19.137", port=6379, db=9)
	# 加载用户特征
    client_of_features = redis.StrictRedis(host="192.168.19.137", port=6379, db=4)
    # 获取用户特征
    user_feature = json.loads(client_of_features.hget("user_features", userId))
    
    # 获取用户召回集
    recall_sets = client_of_recall.smembers(userId)
    
    result = []
    
    # 遍历召回集
    for adgroupId in recall_sets:
        adgroupId = int(adgroupId)
        # 获取该广告的特征值
        ad_feature = json.loads(client_of_features.hget("ad_features", adgroupId))
        
		# 定义字典特征
        features = {}
		# 添加用户特征
        features.update(user_feature)
		# 添加广告特征
        features.update(ad_feature)

		# 遍历特征数据, 如果没有特征值填充为-1
        for k,v in features.items():
            if v is None:
                features[k] = -1

		# 准备特征列表
        features_col = [
            # 特征值
            "price",
            "cms_segid",
            "cms_group_id",
            "final_gender_code",
            "age_level",
            "shopping_level",
            "occupation",
            "pid", 
            "pvalue_level",
            "new_user_class_level"
        ]
        '''
        "cms_group_id", 类别型特征，约13个分类 ==> 13维
        "final_gender_code", 类别型特征，2个分类 ==> 2维
        "age_level", 类别型特征，7个分类 ==>7维
        "shopping_level", 类别型特征，3个分类 ==> 3维
        "occupation", 类别型特征，2个分类 ==> 2维
        '''
		# 把价格转换为小数
        price = float(features["price"])
		
		# 资源位 "430548_1007", "430549_1007"
		# 二维
        pid_value = [0 for i in range(2)]#[0,0]
		# 消费档次 1:低档，2:中档，3:高档； 加上缺失值 =>  四维
        pvalue_level_value = [0 for i in range(4)]# [0,0,0,0]
		# 城市层级: 原4四维 + 加上缺失值 =>  5维 
        new_user_class_level_value = [0 for i in range(5)] # [0,0,0,0,0]
        
		# 利用对用产生one-hot编码后的特征数据
		pid_value[pid_rela[pid]] = 1 #[1,0]
    
    pvalue_level_value[pvalue_level_rela[int(features["pvalue_level"])]] = 1
        new_user_class_level_value[new_user_class_level_rela[int(features["new_user_class_level"])]] = 1

		# 把用户和资源位对应的特征合在一起
        vector = DenseVector([price] + pid_value + [int(features["cms_segid"])]+[int(features["cms_group_id"])] + [int(features["final_gender_code"])]\
        + [int(features["age_level"])] + [int(features["shopping_level"])] +[int(features["occupation"])] + pvalue_level_value + new_user_class_level_value)

        result.append((userId, adgroupId, vector))
        
    return result

# 传入用户ID和广告资源位 返回当前用户的召回集对应的物品特征，以及用户特特征
create_datasets(88, "430548_1007")
```
### 19_加载缓存特征产生推荐结果
- 步骤
	- 加载逻辑回归模型
	- 创建Pandas的dataframe 指定三列 用户id 广告id 特征向量
	- 把数据转换为spark的dataframe
	- 利用逻辑回归的模型 预测物品的点击率 按照probability 从小到大进行排列
	- 选择前10广告进行推荐

- 代码
	```python
	from pyspark.ml.classification import LogisticRegressionModel
	# 加载逻辑回归模型 对所有的商品进行排序
	CTR_model = LogisticRegressionModel.load("/models/CTRModel_Normal.obj")
	#创建Pandas的dataframe 指定三列 用户id 广告id 特征向量
	pdf = pd.DataFrame(create_datasets(8, "430548_1007"), columns=["userId", "adgroupId", "features"])
	datasets = spark.createDataFrame(pdf)
	datasets.show()

	# 利用逻辑回归的模型 预测物品的点击率 按照probability 从小到大进行排列
	prediction = CTR_model.transform(datasets).sort("probability")
	prediction.show()

	# 获取推荐的广告ID
	[i.adgroupId for i in prediction.select("adgroupId").head(10)]
	```

### 20_推荐结果产生回顾[10:11]
- 整体思路
  - 使用ALS训练协同过滤模型, 为用户找到最喜欢的类别
  - 创建逻辑回归模型, 利用用户特征和物品(广告对应商品)价格作为特征, 预测用户是否会点击
  	- 使用随机森林处理缺失值(在本案例中没有用)
  	- 把缺失值当成一个类别, 使用One-Hot处理类别特征; 只适用与类别特征
  	- 选择特征和目标值, 划分方式
  	- 划分测试和训练集
  	- 使用逻辑回归训练模型, 对点击率进行预测
  	- 按照probability 升序排序(预测的没有点击的概率)
  - 缓存数据
  	- 缓存召回结果
  	- 缓存广告特征
  	- 缓存用户特征
  - 加载特征数据, 把真实特征转换为模型需要的特征 
  - 获取 指定用户和资源位的特征数据, 使用逻辑回归进行预测
  - 按照probability进行升序排序, 选择靠前的10个推荐给用户



​	
