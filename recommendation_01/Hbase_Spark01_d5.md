```
转人工智能方向，本以为可以逃离SQL语句。最终还是逃不出它的魔爪。
```



### Hive回顾(15:36)

- Hive是一个数据仓库工具
- Hive是基于Hadoop的， 数据存储在HDFS上，计算依赖mapreduce
- Hive实际上两件事儿①负责元数据的存储（数据库名字，表名字，表中字段名字类型....)②把HQL 翻译成 hdfs命令或者Mapreduce
- Hive和关系型数据库的区别
  - Hive做数据仓库中 大规模数据的离线分析
    - 数据仓库 数据只是增加和查询 hive一般不做修改和删除
    - 大规模数据 T级别的
    - 离线分析  慢
    - HQL 和 SQL有一些区别
    - 数据类型 除了原子类型外还支持复杂类型

- 使用Hive的时候
  - 先启动MySQL 元数据是保存在MYSQL中的
  - 再启动HDFS（数据存储），YARN（调度mapreduce任务的），只有启动了YARN hive复杂查询才能够使用，因为涉及到把HQL翻译成MapReduce并且要指定MR任务，这个时候需要YARN去协调
  - 还要启动元数据服务
    - hive --service metastore&
- Hive的shell命令
  - 创建表的时候指定分割符
  - 加载数据的时候尽量使用load data方式 不要用insert, 因为太慢了
- 内部表外部表
- 分区表
- UDF 相当于map
- UDAF 相当于reduce

### 02_HBase简介[20:43]
- 目标: 知道HBase的特点 
- HBase的特点:
  - 分布式的开源数据库，是BigTable的开源实现
  - **面向列**的数据库
  - 适合**非结构化数据**的存储

- HBase和传统关系型数据库的区别
  - HBase PB级别 , 只有Bytes数据类型, 单行事务，row-key有索引，写入快
- **面向列的数据库**
  - 列存储 一列放在一起 是一个单独的文件
- **非结构化数据**
  - 非结构化数据是数据结构不规则或不完整
  - 没有预定义的数据模型
  - 不方便用数据库二维逻辑表来表现
  - 业务经常变化的数据
  - 视频，图片
  - Mysql处理一些互联网业务，可能会出现数据极度稀疏的情况，这样的业务就适合用HBase来保存，不会出现稀疏的问题，有字段就记下来，没有就不计 
  - 联系方式的保存
    - 固定电话，手机，手机2 ，微信，QQ，微博。。。
    - Contact （联系方式）
      - phone:89867887, cell:13888888, qq:
      - wechat:  weibo



### 03_列存储和非关系型数据[08:40]
- 目标: 
  - 知道HBase与Hive的区别

- HBase在Hadoop生态中地位
  - HBase 数据是保存到HDFS上，但是查询的功能，相关的计算都是由HBASE自己完成的
  - HBase的集群管理依赖于ZooKeeper

- HBase和Hive的区别
  - Hive 离线分析 处理的是结构化数据 join问题是可以处理的, 数据存储在HDFS上, 计算需要MapReduce
  - HBase 实时查询， 处理的是非结构化数据，表之间关系复杂的业务不适合用HBase处理, 数据存储在HDFS上, 自己完成计算, 依赖Zookeeper. 

### 04_HBase的数据模型☆☆☆☆☆[13:30]

- HBase没有数据类型 所有内容都是用bytes形式保存的
- NameSpace 数据库
- table 表
- row  每一行都有一个row key
- row key  行键 在HBase中 每一行一定有一个Row key，而且只有这个Row key 才有索引
- 列  Column family： Column qualifier：value
  - Column family列族
    - 创建表的时候需要指定列族，HBase的表结构只有列族的名字，在创建表的时候，只需要指定表名字和列族的名字
    - 每一个列族单独放在一个文件中，如果数据量比较大的情况下，会出现数据的分割，一个列族可能对应着多个文件
  -  Column qualifier列标识符 
    - 在创建表的时候不需要指定的
    - 可以与关系型数据库的列对应

- 时间戳和数据版本
  - HBase是追加型数据库
  - 如果有更新的操作，HBase不会删除原有的数据，而是把新的数据记录下来，更新一下版本号，会保存同一个单元格的多个版本的数据，多版本的数据可以通过时间戳来指定访问，也可以通过版本号来访问
- HBase只支持行级别事务

### 05_HBase的安装和启动☆☆☆[10:36]
- 目标: 能够启动hbase

- hbase使用内部zookeeper的启动流程
  - 启动Hadoop的HDFS
    - root/bigdata/hadoop/sbin/  start-dfs.sh
- 关闭安全模式
    - hdfs dfsadmin -safemode leave
  - 在任意位置下, 启动hbase 
    -  start-hbase.sh
    - 原因: 已经被加入到环境变量中了.
  
- hbase使用外部zookeeper的启动流程
  - 启动Hadoop的HDFS
    - root/bigdata/hadoop/sbin/  start-dfs.sh
  - 启动zookeeper
    - 在任意位置下 zkServer.sh start
  - 启动HBase
    - 在任意位置下 调用 start-hbase.sh

  - 由于虚拟机暂停,挂起操作后, 会导致虚拟机时间和本地时间不一致, 此时需要对时间进行同步
    - ntpdate cn.pool.ntp.org


### 06_HBase的Shell命令☆☆☆☆[22:26]
目标: 能够使用HBase的Shell命令 操作表 和 对数据进行增删改查
- 操作表
  - 查看表: list
  - 删除表: 
    - 禁用表: disable '表名'
    - 删除表: drop '表名'
  - 创建表
    - create '表名','列族'
  - 查看名称空间: 
    - list_namespace
  - 创建名称空间:
    - create_namespace '名称空间名称'
  - 在名称空间下创建表
    - create '名称空间名称:表名','列族'
  - 查看名称空间下的表
    - list_namespace_tables '名称空间名称'
- 操作数据
  - put '表名','行键','列族:列标识符','值'


- 查询的时候有两种方式
  - scan查询 使用scan的时候需要注意，默认是全表的搜索，由于HBase一般用来支撑大数据的业务，涉及到的数据量比较大的，如果触发全表的搜索，可能等待的时间会比较长，所以使用scan的时候一般要加上限定条件，LIMIT => 10 STARTROW => '起始的rowkey' ENDROW => '结束rowkey'
    ```sql
    scan '名称空间:表名', {COLUMNS => ['列族名1', '列族名2'], LIMIT => 10, STARTROW => '起始的rowkey', ENDROW=>'结束rowkey'}  # 通过COLUMNS  LIMIT STARTROW 等条件缩小查询范围
    -- 通过rowkey的前缀查询
    scan 'user', {ROWPREFIXFILTER=>'rowkey_22'}
    ```

  - get 查询 会用到rowkey的索引
    ```sql
    get 'user','rowkey_16'
    get 'user','rowkey_16','base_info'
    get 'user','rowkey_16','base_info:username'
    get 'user', 'rowkey_16', {COLUMN => ['base_info:username','base_info:sex']}
    ```
  - scan 缩小查询范围 和get 找到指定行的时候都会用到rowkey  只有rowkey是有索引的，所以rowkey的设计在hbase中是比较重要的环节
  - rowkey在保存的时候是按照字典序排列的，而且索引只是记录了某一个region的起始和结束的范围
- 删除数据
  ```sql
  delete 'user', 'rowkey_16', 'base_info:username'
  ```
- 清空数据: truncate 'user'
- 修改表结构
  - 添加列族: alter 'user', NAME => 'f2'
  - 删除列族: alter 'user', delete => 'f2'
- 查看表结构: desc '表名'

### 07_HBase的Shell命令2☆☆☆☆[22:36]
- 目标: 理解Hbase的时间戳和版本号

- 查询多版本数据
  ```sql
  get 'user','rowkey_10',{COLUMN=>'base_info:username',VERSIONS=>2}
  ```

- 修改hbase保存的版本数量
  ```sql
    alter 'user',NAME=>'base_info',VERSIONS=>10
  ```
  
- 通过时间戳查询
  - 通过TIMERANGE 指定时间范围
    ```sql
      scan 'user',{COLUMNS => 'base_info', TIMERANGE => [1558323139732, 1558323139866]}
      get 'user','rowkey_10',{COLUMN=>'base_info:username',VERSIONS=>2,TIMERANGE => [1558323904130, 1558323918954]}
    ```
  - 通过时间戳过滤器 指定具体时间戳的值
    ```sql
      scan 'user',{FILTER => 'TimestampsFilter (1558323139732, 1558323139866)'}
      get 'user','rowkey_10',{COLUMN=>'base_info:username',VERSIONS=>2,FILTER => 'TimestampsFilter (1558323904130, 1558323918954)'}
    ```
  - 获取最近多个版本的数据
    ```sql
      get 'user','rowkey_10',{COLUMN=>'base_info:username',VERSIONS=>10}
    ```
  - 通过指定时间戳获取不同版本的数据
    ```sql
    get 'user','rowkey_10',{COLUMN=>'base_info:username',TIMESTAMP=>1558323904133}
    ```
  
- 时间戳和版本号
  - 保存数据的时候 都会有一个时间戳和版本号
  - 可以通过column family的属性VERSIONS 来控制可以查询到多少个版本
  - 当数据需要更新的时候，HBase没有提供一个update的方法，而是通过再一次put的方式
  - 如果是同一行数据，每put一次 ，版本号+1,所有的数据都会保留下来
  - 可以通过Versions来指定查询几个版本，也可以通过时间戳来查询指定的版本数据

- 注意:

  -  hbase 所有的名称都必须加 ' ', 否则会被当成本地变量,
  - hbase严格区分大小写. 

### 08_HappyBase操作HBase1☆☆☆☆☆

- 目标: 能够创建连接,关闭链接, 创建表, 删除表, 查询表

- 把HBase 的 thrift server启动: hbase-daemon.sh start thrift
- 先获取 Connection
- 通过connection对象可以对表进行操作
  - 创建表 connection.create_table('user2',{'cf1':dict()})
  - 查询表 connection.tables()
  - 删除表 connection.delete_table('user2',True)
- 通过Connection获取table
  
  - 通过table对数据进行操作
 - 查询
    - scan
      ```python
      filter = "ColumnPrefixFilter('username')"
      result = table.scan(row_start='起始rowkey',filter = filter)
      for row_key, row_data in result:
        print(row_key,row_data)
      ```
    - get
      ```python
      result = table.row('rowkey的值',columns=['列族:列标识符'])
      result = table.row('rowkey的值')
      result = table.rows(['rowkey的值','rowkey的值'])
      ```

### 09_HappyBase操作HBase2(10:38)

- 目标: 能够添加数据, 删除数据 

- 插入数据
  - ```python
    table.put('rowkey值',{'列族:列标识符':'值'})
    ```
- 删除数据
  - ```python
    table.delete('rowkey值', ['列族:列标识符'])
    ```

### 10_HBase组件☆☆☆☆[18:39]

- ZK(Zookeeper)
  - 服务的注册与发现，HMaster和HRegionServer都向zk注册地址，客户端向zk请求HMaster和HRegionServer的地址
  - HMaster主节点选举
  - HMaster元数据存储
  - HRegionServer状态汇报
- HMaster
  - 表数据的CRUD
  - HRegionServer Region划分
  - Region负载均衡、监控Region状态
- HRegionServer
  - 数据存储
  - 切分在运行过程中变得过大的region
- HRegionServer->HRegion->HStore（对应一个列族） Memstore，storefile

* Write-Ahead-Log（WAL）保障数据高可用
  * 在每次用户操作写入MemStore的同时，也会写一份数据到HLog文件中（HLog文件格式见后续），HLog文件定期会滚动出新的，并删除旧的文件（已持久化到StoreFile中的数据）

### 11_HBase物理存储原理[11:00]

- Flush
  - 当MemStore满了以后会Flush成一个StoreFile（底层实现是HFile）
- Compact
  - Minor compact
    - 小的相邻的StoreFile合并成一个大的StoreFile，不会处理数据，只是合起来
  - Major compact
    - 除了文件合并，还有很多其它任务，删除过期数据
- Split 
  - 当当单个StoreFile大小超过一定阈值后，会触发Split操作，同时把当前 Region Split成2个

### 12_HBase自动split带来的问题[04:23]
- 自动split带来的问题
  - 当当单个StoreFile大小超过一定阈值后，会触发Split操作，同时把当前 Region Split成2个
  - 父Region会下线, 会出来两个新的Region
  - 旧的任务下线导致mr job 崩溃

- 如何解决: 
  - 尽量避免自动Region Split (hbase-site.xml设置参数)
    ```xml
      <property>
              <!-- 避免自动split  -->
              <name>hbase.hregion.max.filesize</name>
              <!-- 32G <==> 34359738368-->
              <value>34359738368</value>
      </property>
      <property>
              <name>hbase.hregion.memstore.flush.size</name>
              <!-- 512m <=> 536870912 -->
              <value>536870912</value>
      </property>
      <property>
              <name>hbase.hstore.compactionThreshold</name>
              <value>8</value>
      </property>
      <property>
              <!--达到64个时,强制Compact-->
              <name>hbase.hstore.blokingstoreFiles</name>
              <value>64</value>
      </property>
    ```
    
  - max.filesize < flush.size*blokingstoreFiles 以保证region不会被block
  
  - Region的手工维护
    - flush
    - compact
    - Split
    
  - hbase的管理页面: http://192.168.19.137:16010/master-status
  
  - hbase的运维手册: https://www.cnblogs.com/hit-zb/p/9750418.html

### 13_HBase优化策略[04:53]
- RowKey优化

  - Rowkey是按照字典序排列的
  - Rowkey不要过长，（最大长度64K）建议越短越好（最好不要超过16个字节）
  - Rowkey唯一
  - 防止热点问题, 避免使用时序或者单调的递增递减等
    - rowkey散列 将rowkey高位作为散列字段，由程序随机生成，低位放时序、单调数据
- Column优化
- 利用HBase默认排序特点, 将一起访问的数据放到一起一个column family中
  - 列族的名称和列的描述命名尽量简短 保证可读性的前提下 尽可能短
  - 同一张表中ColumnFamily的数量不要超过3个


### 14_HBase内容回顾[07:53]

- 非关系型数据  ☆☆☆☆☆
- 面向列的数据库  ☆☆☆☆☆
- HappyBase操作HBase ☆☆☆☆☆
- HBase的数据模型☆☆☆☆☆
  - 创建表 只需要表名和列族的名字
  - row_key
    - 有索引的
  - HBase 不区分数据类型
- HBase 组件 /存储原理
  - Zookeeper
  - HMaster
  - HRegionServer
- 存储原理
  - 追加型的数据库  HBase的写入性能>mysql
  - Flush、compact 、split

### 15_Spark简介[12:40]
- 目标: 知道Spark优势和作用
- 什么是Spark （spark+hdfs)
  - 基于内存的分布式计算框架
- 为什么要学Spark
  - 速度快，api丰富 （对比mr)
  - 生态丰富，提供多种场景的一站式解决方案
  - 易用性 pyspark 官方自带的Python api
  - 兼容性 standalone/yarn/mesos

### 16_Spark的local模式wordcount案例[05:56]
- 启动: hdfs和yarn
- 运行: pyspark
- wordcount 代码:
  ```py
    words = sc.textFile('file:///root/tmp/test.txt') \
              .flatMap(lambda line: line.split(" ")) \
              .map(lambda x: (x, 1)) \
              .reduceByKey(lambda a, b: a + b).collect()
    print(words)
  ```

### 17_RDD的概念[11:09]
- 目标: 理解RDD特点

- 弹性分布式数据集，是Spark中最核心的数据模型
- 弹性：可以在磁盘，也可以在内存，分布式存储也是弹性的
- 不可变：RDD1 变换到RDD2 RDD1是不会变化的
  - 所以有父RDD和子RDD的概念
- 可分区
  - 在创建RDD的时候可以执行RDD分区的数量
  - 每一个分区都对应着一个task，每一个task就由一个线程来计算

### 18_RDD的创建[10:44]
- 目标: 能够创建RDD

- ①创建一个sparkContext
  - 可以通过SparkCotext('master地址','应用名字')
  - 如果打开了pyspark 的shell终端，会自动帮助创建一个sparkcontext

- ②通过sparkContext创建RDD
  - 可以从内存的数据中创建

    ```python
      sc.parallelize(data)
    ```

  - 可以从外部文件创建

    ```shell
       sc.textFile('file:///root/tmp/test.txt')
    ```

- 注意：创建RDD的时候可以指定分区的数量
  - 分区数量决定了Spark作业的task数量，一个task对应一个分区
  - 分区数量决定了运行spark作业的时候线程数量
  - 一个CPU内核对应2~4个spark 的task 对应2~4个分区