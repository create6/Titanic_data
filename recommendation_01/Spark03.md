```
看到RDD,就立马想到了DDR。想电脑已经想疯了！！现学的几个框架 在工作中那几个最常用？
在推荐系统中: 最常用的 HDFS 和 Spark
在数据仓库中: 最常用的 HDFS 和 Hive
```



### 01_内容回顾[16:38]

- RDD
  - 弹性分布式数据集
  - 不可变
  - 可分区

- Spark SQL
  - 处理结构化的数据
  - 作用Hive类似
  - 优势 代码少 速度快
- DataFrame
  - 特殊的RDD，带着schema的RDD
  - 创建DataFrame 首先要有 sparkSession
  - 通过SparkSession 再创建dataframe
  - createDataFrame
  - .read.XXX

### 02_SparkSQL案例_数据去重[12:46]
- 目标: 能够对数据进行去重

- 数据去重
  
  ```python
  #删除完全重复的数据
  df.dropDuplicates()
  # 可以指定哪些字段一样算是重复
  df2.dropDuplicates(subset = ）
    
  # 有意义的重复记录去重之后，再看某个无意义字段的值是否有重复（在这个例子中，是看id是否重复）
  # 查看某一列是否有重复值
  import pyspark.sql.functions as fn
    df3.agg(fn.count('id').alias('id_count'),fn.countDistinct('id').alias('distinct_id_count')).collect()
    
  # 4.对于id这种无意义的列重复，添加另外一列自增id
  df3.withColumn('new_id',fn.monotonically_increasing_id()).show()
  ```


### 03_SparkSQL案例_缺失值处理[20:46]
- 目标: 能够处理缺失值

- 缺失值处理
  - 查看数据的缺失情况 

    ```python
    # 1.计算每条记录的缺失值情况
    df_miss.rdd.map(lambda row:(row['id'],sum([c==None for c in row]))).collect()
    ```

  - 如果缺失比较严重的可以考虑删除
    
    ```python
    # 3、删除缺失值过于严重的列, 其实是先建一个DF，不要缺失值的列
    df_miss_no_income = df_miss.select([c for c in df_miss.columns if c != 'income'])
    ```

  - 按行删除缺失严重的数据
    ```python
    df_miss_no_income.dropna(thresh=3).show()
    ```
    - thresh 阈值, 缺失值达到指定数量的行删除. 

  - 缺失值的填充
    - 如果值是连续的，那么可以使用统计学方式，计算平均值，中位数进行填充
    - 用算法预测缺失值
    - 如果是分类的值，可以使用默认值填充、也可以用算法来预测
  - spark的dataframe 填充缺失值
    - dataframe.fillna()
    - 可以传入一个字符串，数字，当前dataframe所有缺失的值都用相同的替代
    - 也可以传入一个dict, {列名：要填充的缺失值）为每一列指定不同的默认值
      ```python
      # 先计算均值，并组织成一个字典
      means = df_miss_no_income.agg(*[fn.mean(c).alias(c) for c in df_miss_no_income.columns if c != 'gender']).toPandas().to_dict('records')[0]
      # 然后添加其它的列
      means['gender'] = 'missing'
      df_miss_no_income.fillna(means).show()
      ```

### 04_SparkSQL案例_异常值处理[13:31]
- 目标: 能够处理异常值

- 异常值处理
  - 找到数据中 过大或者过小的值，确定下来可以接受的数据下限和数据上限，
  - 如果某个值小于下限，就用下限值去替换，如果某个值大于上限，就用上限的值替换
  - 分位数去极值


- 分位数:
  - API: df.approxQuantile(col, probabilities, relativeError) 
  - 分数:
    - col: 列名
    - probabilities: 想要计算的分位点，可以是一个点，也可以是一个列表（0和1之间的小数）
    - relativeError: 第三个参数是能容忍的误差，如果是0，代表百分百精确计算。
      - 分为点范围: floor((p - err) * N) <= rank(x) <= ceil((p + err) * N).
  - 返回分位点列表



### 05_SparkStreaming简介[08:24]
- 目标: 说出SparkStreaming的特点

- SparkStreaming的特点

  - 准实时的 计算框架，延迟比较低，1S左右
  - 采用micro-batch 处理数据
  - 指定一个时间间隔，每隔相同的一段时间就到数据源取数据
  - 对比storm生态丰富，python支持更友好



### 06_SparkStreaming的wordcount案例[24:26]
- 目标:
  - 说出 Streaming Context 与 DStream 离散流 的特点
  - 能够使用SparkStreaming的wordcount案例
- SparkStreaming的组件
  - Streaming Context
    - 流式计算的上下文
    - 通过spark context创建 streaming context
  - DStream 离散流
    - Sparkstreaming 数据抽象
    - 由一系列的RDD组成
  - 关于Streaming Context几点说明

    ```python
      #参数1 spark context,参数2 获取数据时间间隔
      StreamingContext(sc, 1)
    ```

    - Streaming Context 使用套路
      - 先创建Streaming Context，
      - 写处理DStream数据的逻辑，
      - 逻辑写完调用Streaming Context的start方法 开启流式计算

    - 一旦Streaming Context调用了stop 就不能重新启动，这个对象会被删除，
    - 如果想接着算，需要创建一个新的Streaming Context

    - 一个SparkContext 只能创建一个StreamingContext
    
    - StreamingContext关闭的时候默认会把 spark context也关掉，`sc.stop(False)` 只会关闭StreamingContext

  - 步骤:
    - 先创建StreamingContext
    - 写对DStream的处理，实际上处理的API和RDD基本一样的 
    - 调用StreamingContext的start


- Spark Streaming编码步骤：
  1. 创建一个StreamingContext
  2. 从StreamingContext中创建一个数据对象
  3. 对数据对象进行Transformations操作
  4. 输出结果
  5. 开始和停止

- 利用Spark Streaming实现WordCount

- 需求：监听某个端口上的网络数据，实时统计出现的不同单词个数。
  1. 需要安装一个nc工具：sudo yum install -y nc
  2. 执行指令：nc -lk 9999 -v

- DStream常用API

  * 获取监听在某个端口上DStream

    ```python
    sc = SparkContext("local[2]",appName="NetworkWordCount")
    #参数2：指定执行计算的时间间隔
    ssc = StreamingContext(sc, 1)
    #监听ip，端口上的上的数据
    lines = ssc.socketTextStream('localhost',9999)
    ```

  * flatMap, map, reduceByKey 等RDD的transform相关的方法

  * pprint: 把结果打印到控制台

- 代码

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

    from pyspark import SparkContext
    from pyspark.streaming import StreamingContext

    if __name__ == "__main__":

        sc = SparkContext("local[2]",appName="NetworkWordCount")
        #参数2：指定执行计算的时间间隔
        ssc = StreamingContext(sc, 1)
        #监听ip，端口上的上的数据
        lines = ssc.socketTextStream('localhost',9999)
        #将数据按空格进行拆分为多个单词
        words = lines.flatMap(lambda line: line.split(" "))
        #将单词转换为(单词，1)的形式
        pairs = words.map(lambda word:(word,1))
        #统计单词个数
        wordCounts = pairs.reduceByKey(lambda x,y:x+y)
        #打印结果信息，会使得前面的transformation操作执行
        wordCounts.pprint()
        #启动StreamingContext
        ssc.start()
        #等待计算结束
        ssc.awaitTermination()
  ```

### 07_SparkStreaming的状态操作updatestatebykey[14:03]
- 目标: 应用updatestatebykey

- 在Spark Streaming中存在两种状态操作
  - UpdateStateByKey
  - Windows操作
- 使用有状态的transformation，需要开启Checkpoint
  - spark streaming 的容错机制
  - 它将足够多的信息checkpoint到某些具备容错性的存储系统如hdfs上，以便出错时能够迅速恢复

- updateStateByKey: 会保留streamingcontext启动之后的所有结果
  - 需要设置一个checkpoint目录，用来在磁盘上缓存对应的记录，避免出现节点崩溃的时候，数据的丢失
  - 调用updateStateByKey
  
    ```python
      updateStateByKey（新的数据，旧的结果）

      #定义state更新函数
      def updateFunc(new_values, last_sum):
          return sum(new_values) + (last_sum or 0)

      lines = ssc.socketTextStream("localhost", 9999)
      # 对数据以空格进行拆分，分为多个单词
      counts = lines.flatMap(lambda line: line.split(" ")) \
          .map(lambda word: (word, 1)) \
          .updateStateByKey(updateFunc=updateFunc) # 应用updateStateByKey函数
    ```




### 08_SparkStreaming的状态操作_窗口操作[18:04]
- windows
  - 统计一个时间窗口内的数据, 比如: 5分钟的热搜
  - 需要指定窗口长度，滑动的时间间隔
    - 注意：窗口长度，滑动的时间间隔，必须是创建StreamingContext的时候，指定取数据的时间间隔的整数倍
  - reduceByKeyAndWindow(func,invFunc,windowLength,slideInterval)
    - 参数
    - func: 一个用来处理窗口滑动时，新滑入的数据
    - invFunc: 另外一个用来处理窗口滑动时，滑出去的数据
    - windowLength: 窗口长度
    - slideInterval: 窗口滑动的时间间隔
  
- 代码
  ```python
    # 定义基本的时间间隔, 取数, 窗口长度, 滑动窗口的时间间隔都以这个基本单位为准, 需要是它的整数倍
    batch_interval = 1  # base time unit (in seconds)
    # 定义窗口长度
    window_length = 6 * batch_interval
    # 定义滑动时间间隔
    frequency = 3 * batch_interval

    #1指的是, 每隔1s钟会到数据源获取一次数据
    ssc = StreamingContext(sc, batch_interval)
    # 设置检查点
    ssc.checkpoint('/checkpoint')

    # 根据数据源, 选择合适的方法, 创建数据连接, 获取数据
    lines = ssc.socketTextStream('localhost', 9999)
    addFunc = lambda x,y: x + y
    invFunc = lambda x, y: x - y
    window_counts = lines.map(get_countryname).reduceByKeyAndWindow(addFunc, invFunc, window_length, frequency)
  ```

### 09_StreamingContext的停止问题[06:14]
- 总结 和 问题说明

- Streaming Context
  - 在Streaming Context上调用Stop方法, 也会关闭SparkContext对象, 如果只想仅关闭Streaming Context对象,设置stop()的可选参数为false
  - 一个SparkContext对象可以重复利用去创建多个Streaming Context对象(不关闭SparkContext前提下), 但是需要关一个再开下一个


### 10_lambda框架主要组件回顾[17:19]

- Hadoop
  - **HDFS** 分布式文件系统
  - MapReduce
- Spark  分布式的计算框架（基于内存）

  - Spark core
  - Spark SQL
  - Spark Streaming

- Hive

  - 数据仓库的工具
  - HQL翻译成 MapReduce或者hdfs命令
  - 使用场景
    - (数据分析，大数据相关-数仓 etl）
    - 在推荐业务中 更多还是提供一个元数据服务

- HBase
  - 开源的分布式数据库
  - 面向列，存非关系型数据，nosql数据库
    - 面向列： 每一个column family是一个文件
    - 非关系型数据：
    - key value型的数据库
  - happybase操作HBase
- Zookeeper
  - 分布式集群管理工具
  - 服务的注册与发现
  - 主节点选举
  - 节点内容一致性保证

- Flume: 日志收集
- Kafka: 消息队列
- sqoop: 关系型数据库数据与HDFS的数据交互

### 11_lambda框架主要组件回顾2[28:24]
- 对各个知识点进行细化

- Hadoop

  - **HDFS** 分布式文件系统
    - namenode
      - 元数据存储
      - datanode管理
      - 响应客户端的请求返回元数据信息
    - datanode
      - 数据的读写
      - 自身节点数据的管理
      - 定期向namenode汇报状态
    - 特点
      - 数据拆分成block保存：128MB
      - 数据冗余 3个副本
  - MapReduce
    - 分布式的计算框架
    - 编程的思想
      - map
      - reduce
    - mrjob
    - map reduce之间有一次shuffle ，shuffle是在磁盘上完成的

- Spark  分布式的计算框架（基于内存）

  - Spark core
    - spark context
    - RDD
      - transformation
      - action
      - persist
  - Spark SQL
    - spark session
    - DataFrame
  - Spark Streaming
    - streaming context
    - DStream

- Hive

  - 数据仓库的工具

  - HQL翻译成 MapReduce或者hdfs命令

  - 使用场景（数据分析，大数据相关-数仓 etl）

    在推荐业务中 更多还是提供一个元数据服务

  - 结构

    - 用户接口  命令行
    - 元数据存储  mysql
    - 驱动（HQL翻译成 MapReduce或者hdfs命令)

  - 数据模型

    - 数据库 表 分区表： 文件夹
    - 数据：文件夹里的文件

  - 数据类型: hive支持复杂数据类型 array map struct

  - 内部表 外部表 分区表

  - UDF/UDAF

  - explode/lateral view explode  collect_set/collect_list contract/contract_ws str_to_map

- HBase

  - 开源的分布式数据库
  - 面向列，存非关系型数据，nosql数据库
    - 面向列： 每一个column family是一个文件
    - 非关系型数据：
    - key value型的数据库
  - happybase操作HBase
    - connection
      - 可以对表做管理
    - conncetion->table
      - 通过table可以对数据进行管理
    - 查询
      - scan
        - 全表搜索  避免
        - startrow 指定rowkey范围 endrow
      - get
        - 先命中rowkey

  - HBase的数据模型

    - 不区分数据类型，byte
    - namespace 
    - 创建表的时候 只需要指定表名，列族名字
    - 行 row_key
    - 列族  cf:cq:value
    - 列标识符
    - 时间戳、版本号

  - HBase的集群角色

    - HMaster

    - HRegionServer

      - HRegion
      - HStore
        - storefile
        - memstore
    - Zookeeper


  - HBase数据的 flush compact split 操作
    - 避免HBase自动compact split

- Zookeeper
  - 分布式集群管理工具
  - 服务的注册与发现
  - 主节点选举
  - 节点内容一致性保证


- Flume: 日志收集
- Kafka: 消息队列
- sqoop: 关系型数据库数据与HDFS的数据交互

### 12_推荐算法回顾[09:47]

- 信息过滤系统
- 信息过载 用户需求不明确

- 召回  过滤调整 排序  过滤调整
- 召回  recall
  - 协同过滤
    - user item matrix
    - 用户对物品的评分矩阵
    - 如果用户对物品的评分数据稠密
      - 基于用户的协同过滤
        - 通过用户的消费记录 找到相似用户
        - 构建用户向量 计算向量之间相似度
      - 基于物品的协同过滤
        - 通过用户的消费记录 找到相似物品
        - 构建物品向量 计算向量之间相似度
    - 如果用户对物品的评分数据稀疏
      - Model Based CF
        - BaseLine
        - 矩阵分解 LFM
          - 大矩阵拆分成两个小矩阵
          - 隐因子的个数 <<之前矩阵的维度
  - 基于内容
- 排序 ranking
  - LR
    - 用户的特征+物品的特征  用户点/没点 作为目标

### 13_推荐算法回顾2[04:43]

- 召回  recall
  - 基于内容
    - 物品画像
      - tf-idf 
      - 建立 标签-》物品
    - 如果有用户行为数据
      - 利用用户消费过的物品对应的关键词建立用户画像
      - 用户画像的标签-》物品
      - 利用用户消费的次数，用户的评分
    - 没用用户行为数据
      - 标签-> 词向量
      - 物品(多个标签) -> 文档向量
      - 计算向量的相似度找到相似物品
- 排序 ranking
  - LR
    - 用户的特征+物品的特征  用户点/没点 作为目标


### 14_电商推荐案例数据介绍[16:48]
- 目标: 知道电商项目都用到哪些数据集

- 原始样本骨架raw_sample
  - user_id：脱敏过的用户ID；
  - adgroup_id：脱敏过的广告单元ID；
  - time_stamp：时间戳；
  - pid：资源位；
  - noclk：为1代表没有点击；为0代表点击；
  - clk：为0代表没有点击；为1代表点击；

- 广告基本信息表ad_feature
  - adgroup_id：脱敏过的广告ID；
  - cate_id：脱敏过的商品类目ID；
  - campaign_id：脱敏过的广告计划ID；
  - customer_id: 脱敏过的广告主ID；
  - brand_id：脱敏过的品牌ID；
  - price: 宝贝的价格
- 用户基本信息表user_profile
  - userid：脱敏过的用户ID；
  - cms_segid：微群ID；
  - cms_group_id：cms_group_id；
  - final_gender_code：性别 1:男,2:女；
  - age_level：年龄层次； 1234
  - pvalue_level：消费档次，1:低档，2:中档，3:高档；
  - shopping_level：购物深度，1:浅层用户,2:中度用户,3:深度用户
  - occupation：是否大学生 ，1:是,0:否
  - new_user_class_level：城市层级
- 用户的行为日志behavior_log
  - user：脱敏过的用户ID；
  - time_stamp：时间戳； 
  - btag：行为类型, 包括以下四种类型 
    - pv  浏览 ​
    - cart 加入购物车 ​ 
    - fav  喜欢 
    - buy  购买
  - cate_id 脱敏过的商品类目id； 
  - brand_id: 脱敏过的品牌id；
- 数据集的下载地址: https://tianchi.aliyun.com/dataset/dataDetail?dataId=56

### 15_模型构建思路介绍[08:14]
- 理解模型构建的思路

- 推荐业务处理主要流程： 召回 ===> 排序 ===> 过滤
  - 离线处理业务流
    - raw_sample.csv + ad_feature.csv + user_profile.csv ==> CTR点击率预测模型
    - behavior_log.csv ==> 评分数据 ==> user-cate/brand评分数据 ==> 协同过滤 ==> top-N cate/brand ==> 关联广告
    - 协同过滤召回 ==> top-N cate/brand ==> 关联对应的广告完成召回
  - 在线处理业务流
    - 数据处理部分：
      - 实时行为日志 ==> 实时特征 ==> 缓存
      - 实时行为日志 ==> 实时商品类别/品牌 ==> 实时广告召回集 ==> 缓存
    - 推荐任务部分：
      - CTR点击率预测模型 + 广告/用户特征(缓存) + 对应的召回集(缓存) ==> 点击率排序 ==> top-N 广告推荐结果

- 涉及技术：
  - Flume：日志数据收集
  - Kafka：实时日志数据处理队列
  - HDFS：存储数据
  - Spark SQL：离线处理
  - Spark ML：模型训练
  - Redis：缓存

### 16_CTR预估概念介绍[06:50]

- 计算出用户点击物品的概率
- 在推荐中 传入用户的ID，根据用户ID找到召回集，根据用户的特征和召回集的物品特征计算每一件商品的点击率
- 根据点击率的高低进行排序

### 17_处理思路介绍[03:44]

- 先训练召回模型
- 再训练排序的模型
  - spark中如何训练逻辑回归的模型
- 在训练模型的时候介绍缺失值处理
  - 随机森林预测缺失值（Spark ML spark MLlib 如何使用）
  - one-hot 在spark中 如何进行onehot的处理

- 数据缓存
  - 数据换存到redis
- 实时推荐
  - 训练好的模型加载, 结合实时数据进行推荐



### 学习目标回顾

* 独立实现Spark SQL 进行数据清洗

  * 去重
  * 缺失值的处理
  * 异常值的处理

* 说出SparkStreaming的特点

  * 准时实时分布式计算 
  * Min-batch 下批量
  * 只能启动一次, 只能stop一次; 默认SparkStreaming停止会关闭SparkContext, 如果不想关闭SparkContext, 那么stop(False)

*  说明DStreaming的常见操作API

  * 获取监听指定端口的DStreaming

    ```python
    #监听ip，端口上的上的数据
    lines = ssc.socketTextStream('localhost',9999)
    ```

  * flatmap

  * map

  * reduceByKey
  
  * pprint()

能够应用Spark Streaming实现实时数据处理

能够应用Spark Streaming的状态操作实现词频统计X



```广告点击数据下载地址
https://tianchi.aliyun.com/dataset/dataDetail?dataId=56
```



