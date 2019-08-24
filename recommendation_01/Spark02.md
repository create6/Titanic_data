### Spark

- 分布式的计算框架
- 速度快，api更丰富，功能多，易用性强

### RDD

- spark 最基础的数据结构
- 弹性分布式数据集
  - 弹性
  - 不可变
    - RDD1 变换到RDD2 
  - 可分区
    - 创建RDD的时候可以指定partition的数量
    - partition的数量可以根据申请的CPU内核数决定
    - 一个partition对应一个task,一个task对应一个线程
    - 1一个内核对应2~4个task对应2~4个partition
- 创建RDD
  - spark context
  - sc

### RDD的三类算子

-  transformation

  - 处理rdd之间变换的
  - 延迟执行，不调用action类算子，不会得到结果

-  action 

  - 获取RDD里面的内容

- persist

  - 负责数据存储的算子

- 总结

  - 并行计算的时候，需要注意，有些情况下，结果可能跟预想的有区别

    - ```python
      rdd1 = sc.parallelize([1,2,3,4,5])
      rdd1.reduce(lambda x,y : x-y)
      ```

    - 上面操作的结果会跟RDD的partition数量有关

  - 调用了transformation之后不要忘记要调用action,否则计算不会被执行

  - action中的collect操作要谨慎使用



### ip地理位置统计

- mapPartitions
  - 数据分成多份，一份一份处理，而不是一条一条处理
  - 如果在数据处理过程中，需要连接其他资源，做耗时操作（连数据库），推荐使用mapPartitions
- 广播变量
  - 可以通过广播变量，把多个task都要用到的数据，声明成当前节点的共享数据
  - 好处可以避免数据的多次复制，节省内存

### Spark Standalone模式介绍

- 启动

  - ```shell
    ./start-master.sh -h 192.168.19.137
    ./start-slave.sh spark://192.168.19.137:7077
    ```

- 集群角色
  - Application
  - Master 集群的管理者
    - 接受作业 下发给slave
    - 监控 slave节点状态
    - slave节点如果出错调度任务到其它节点
  - Worker 负责具体计算的节点
    - 监控自身的健康
    - 响应Master下发的作业
    - 启动 Driver和Executor执行作业
  - Driver Executor

- 作业相关概念
  - stage 作业会被划分成1~n个阶段
    - 窄依赖
      - 父RDD的一个partition只指向子RDD的一个partition
    - 宽依赖
      - 子RDD的每一个partition都依赖于父RDD的所有partition
    - stage划分依据
      - 从RDD1变换到RDD2如果是窄依赖属于同一个stage
      - 只要遇到了宽依赖，当前的stage结束
  - DAGScheduler：stage的划分由DAGScheduler 完成的，每个Stage根据RDD的Partition个数决定Task的个数
  - TaskScheduler 把具体的task调度到Executor上执行

### Spark SQL

- spark sql 作用
  - **处理的是结构化的数据**
  - SQL 翻译成 RDD
  - DataFrameAPI  翻译成 RDD
- spark sql 优势
  - SQL/DataFrame 代码更少
  - SQL/DataFrame 运行速度快
- Spark SQL的特性
  - 跟Spark Core 无缝兼容
  - 连接各种数据源使用统一的方式
  - 跟Hive兼容很好，可以使用HQL，可以读取hive的元数据信息
  - 可以把spark sql处理好的数据通过标准的接口（JDBC or ODBC） 暴露出去



#### Spark SQL 的DataFrame

- RDD 的特性 DataFrame都有
  - 不可变
  - tranformation 延迟执行
  - 分布式

- DataFrame是带着schema的RDD，比RDD的API更丰富，执行效率比直接写RDD更高
- 对比pandas 的 DataFrame
  - 可以处理分布式的数据，处理的数据量更大
  - api没有pandas DataFrame丰富

#### DataFrame的创建

- 先要有一个SparkSession

  ```python
  import os
  # 配置spark driver和pyspark运行时，所使用的python解释器路径
  PYSPARK_PYTHON = "/miniconda2/envs/py365/bin/python"
  JAVA_HOME='/root/bigdata/jdk1.8.0_191'
  # 当存在多个版本时，不指定很可能会导致出错
  os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON
  os.environ["PYSPARK_DRIVER_PYTHON"] = PYSPARK_PYTHON
  os.environ['JAVA_HOME']=JAVA_HOME
  # spark配置信息
  from pyspark import SparkConf
  from pyspark.sql import SparkSession
  
  SPARK_APP_NAME = "SparkSQLTest"
  SPARK_URL = "spark://192.168.2.137:7077"
  
  conf = SparkConf()    # 创建spark config对象
  config = (
  	("spark.app.name", SPARK_APP_NAME),    # 设置启动的spark的app名称，没有提供，将随机产生一个名称
  	("spark.executor.memory", "6g"),    # 设置该app启动时占用的内存用量，默认1g
  	("spark.master", SPARK_URL),    # spark master的地址
      ("spark.executor.cores", "4"),    # 设置spark executor使用的CPU核心数
  )
  # 查看更详细配置及说明：https://spark.apache.org/docs/latest/configuration.html
  
  conf.setAll(config)
  
  # 利用config对象，创建spark session
  spark = SparkSession.builder.config(conf=conf).getOrCreate()
  ```

- 通过SparkSession可以创建DataFrame

  - sparkSession.createDataFrame
  - sparkSession.read.XXX

- DataFrame的api

  - DataFrame之间变换的操作 transformation
  - 获取DataFrame数据的 action
  - 其它基本操作

#### DataFrame操作CSV和Json数据

- 加载csv/json
  - spark.read.csv(文件路径)
  - spark.read.json(文件路径)
  - spark.read.format('文件的格式').load(文件路径)
- 加载文件之后 会自动推断文件的schema
- 也可以自己创建schema做修改
  - StructType()
  - 可以创建StructType对象之后add每一个字段
  - 也可以在创建StructType的时候，直接传入一个StructField("id", StringType（））的list
- spark.read.schema(jsonSchema).json()
- spark SQL udf的套路
  - 创建一个方法
  - 调用udf函数传入 方法名字 参数类型 创建一个udf对象
  - 调用这个udf对象 ，而不是直接使用自己创建的方法

- DataFrame数据.rdd 可以转换成RDD去操作，经过RDD的算子处理之后，可以在通过toDF转换回DataFrame



### 重点

RDD的transformation 和 action算子的使用

DataFrame 和 RDD之间的关系

DataFrame具体的使用