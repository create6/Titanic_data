### 01_内容回顾[13:17]

- 分布式计算  MapReduce
- 分布式存储  HDFS
- 所有的大数据框架 
  - 可靠
  - 可扩展

- HDFS MapReduce YARN

- HDFS结构
  - NameNode
  - DataNode
- YARN
  - Resource Manager
  - Node Manager-》Container
  - Application Master
- MapReduce
  - 分布式的存储 导致了分布式的计算
  - 移动计算比移动数据更划算
  - 既是框架 又是一种编程的思想
  - 把复杂的任务 拆分
    - Map
    - Reduce
  - MRJob -》Hadoop Streaming.jar
- Hadoop生态 发行版 读写流程
  - Apache   CDH
  - 读写流程
    - 数据拆分成128MB一块 3副本冗余
    - 写数据的时候客户端负责数据的拆分
    - NameNode 负责提供可以存数据DataNode列表
    - 客户端向列表中的第一台DataNode提交数据和列表
    - DataNode内部按列表复制数据

- 关防火墙 systemctl stop firewalld
- 退出安全模式 hdfs dfsadmin -safemode leave

### 02_Hive的基本概念[10:06]
- 目标: 知道hive的作用

- 基于Hadoop  底层数据是存储在 HDFS 上
- 数据仓库工具
  - 历史数据版本的保存 存和查 基本不做更新 很少删除
  - 能够处理的数据量比较大的
- 结构化的数据
  - mysql能够处理的就是结构化数据
- 映射为一张数据库表
  - 数据不是以表的形式保存的
- HQL(Hive SQL)查询
- Hive的本质
  - 基于Hadoop，数据是存在HDFS 上
  - Hive做的事儿就是把 SQL翻译成MapReduce 或者是 HDFS命令

- Hive用途
  - 用来做离线数据分析 数仓 ETL



### 03_Hive组件

- 目标: 知道Hive包含哪些组件及其作用 

- 用户接口
  
  - 主要用命令行写HQL
  
- 元数据存储
  - 保存 数据库/表的相关信息
    - 表名 、字段名字 类型、数据存储在hdfs上的位置 。。。。
  
- 驱动
  - 解释器、编译器、优化器、执行器
  - 将HQL 翻译成 MR任务/hdfs命令

- Hive和Hadoop
  - Hive 实际上相当于 Hadoop的客户端
  - 负责把作业提交给Hadoop集群的主节点
  - 不需要有集群 也没有hive集群的概念
  
  

### 04_Hive和关系型数据库的区别[09:36]

- 目标: 知道Hive和关系型数据库的区别
- Hive 和关系型数据库的区别
  - hive 使用HQL 进行 海量数据的离线查询（慢），只做查询和保存
  - 关系型数据库 使用SQL进行 业务数据的CRUD 可以支持线上业务
  - 数据类型 
    - 关系型数据库只支持原子数据类型
    - Hive  原子数据类型 复杂数据类型（Array，map，struct）

### 05_Hive的数据模型和安装部署[09:33]
- Hive的数据模型
  - 数据库  就是文件夹 /user/hive/warehouse/
  - 表 在数据库文件夹下的子文件夹
  - 外部表 可以在HDFS上的任何位置
  - 分区表 表的文件夹下的子文件夹
  - 数据  表文件夹下的文件
  
- Hive的安装
  - Hive 安装前需要安装好 JDK 和 Hadoop。配置好环境变量。
  - 下载Hive的安装包 http://archive.cloudera.com/cdh5/cdh/5/ 并解压
  - 进入到 解压后的hive目录 找到 conf目录, 修改配置文件
    ```shell
      cp hive-env.sh.template hive-env.sh
      vi hive-env.sh
    ```
    在hive-env.sh中指定hadoop的路径
      ```
      HADOOP_HOME=/root/bigdata/hadoop
      ```
  - 配置环境变量
  - 配置MySQL元数据存储
  
- hive 启动
  
  - 关闭防火墙
    
    - systemctl stop firewalld   关闭防火墙, 每次虚拟机启动都需要, 因为防火墙每次开机会自动启动
    - systemctl disable firewalld 禁用防火墙, 防火墙开机就不会自启了, 只要执行一次即可
    
  - 启动 hdfs 和 yarn
    
    ```shell
    cd /root/bigdata/hadoop/sbin
    ./start-dfs.sh 
    ./start-yarn.sh
    ```
    
  - 退出 hdfs的安全模式, 由于我们虚拟机缺失数据文件了, 导致在安全模式下无法对hdfs上的数据进行更改
  
    ```shell
    hdfs dfsadmin -safemode leave
    ```
  
    
  
  - 启动docker
  
    > service docker start
  
  - 通过docker 启动mysql
  
    > docker start mysql
  
  - 启动 hive的metastore元数据服务
  
    > hive --service metastore
  
  - 启动hive
  
    > hive
  
- MySQL
  - 用户名：root
  - 密码：password

### 06_Hive的基本使用[19:38]

- 开始有一个hive的小结. 
- 目标
  
  - 知道hive的创建数据库, 使用数据库, 创建表, 向表中加载数据, 查询数据
- 基本使用
  - 创建数据库: `CREATE DATABASE 数据库名;`
  - 查看数据库: `SHOW DATABASES;`
  - 使用数据库: `use 数据库名;`
  - 创建表: 
    ```sql
      CREATE TABLE 表名(字段名 数据类型, 字段名 数据类型, .. ) row format delimited fields terminated by '分割符';
    ```
     - 分隔符 只能是一个字符，不要指定错误，默认是'/001'
  - 向表中加载数据:
    ```sql
      load data local inpath '/home/hadoop/tmp/student.txt' overwrite into table student;
    ```
  - 查询数据
    ```sql
      select * from student;
    ```
  - 分组查询统计, HQL会转换为MapReduce
    ```sql
      select classNo,count(score) from student where score>=60 group by classNo;
    ```

- 注意
  - 创建表的时候 需要指定分隔符
    - 分隔符 只能是一个字符，不要指定错误，默认是'/001'
  - 向表中加载数据，
    - 可以使用load data ， 
    - 也可以直接把数据文件 通过hadoop fs -put的方式copy到表对应的文件夹下
  - 加载数据的时候 最好使用copy文件的方式，最好不用insert 虽说支持
  - 小批量的数据 不适合用hive处理，mapreduce启动比较慢


### 07_Hive的内部表和外部表[08:55]
- 目标: 
  - 知道如何创建外部表 
  - 知道外部表和内部表的区别

- 创建外部表 
  ```sql
    CREATE EXTERNAL TABLE student2 (classNo string, stuNo string, score int) row format delimited fields terminated by ',' location '/tmp/student';
  ```

- 外部表和内部表区别
  - 创建表的时候 外部表需要添加 EXTERNAL关键字，还需要通过 `location` 指定数据在HDFS上的位置
  - 删除表的时候 外部表只会删除元数据，数据不会被删除，内部表会把数据和元数据全部删除掉

### 08_Hive的分区表[11:18]
- 目标:
  - 知道什么是分区表和其作用
  - 能够创建和使用分区表

- 分区表
  
  - 就是把一个大表划分为若干个小表
- 作用
  
- 优化查询, 查询时尽量利用分区字段。如果不使用分区字段，就会全部扫描。
  
- 使用
  - 创建分区表
    ```sql
    create table employee (name string,salary bigint) partitioned by (date1 string) row format delimited fields terminated by ',' lines terminated by '\n' stored as textfile;
    ```
  - 查看分区
    ```sql
      show partitions employee;
    ```
  - 添加分区
    ```sql
    alter table 表名 add if not exists partition(分区字段名='分区字段的值');
    -- 举例
    alter table employee add if not exists partition(date1='2018-12-01');
    ```
  - 加载数据到分区
    ```sql
    load data local inpath '/root/tmp/employee.txt' into table employee partition(date1='2018-12-01');
    ```
  - 注意
    - 如果重复加载同名文件，不会报错，会自动创建一个*_copy_1.txt
    - 外部分区表即使有分区的目录结构, 也必须要通过hql添加分区, 才能看到相应的数据
    - 如果查询的时候 可以使用分区字段，尽量利用

### 09_Hive分区表2
- 目标:
  - 知道hive加载同名文件会怎样
  - 知道什么是动态分区以及注意事项

- 如果重复加载同名文件，不会报错，会自动创建一个*_copy_1.txt
  - 注意: 此时加载语句中不能有overwrite, 因为overwrite是覆盖原数据

- 动态分区: 在写入数据时自动创建分区(包括目录结构)
  - 注意: 需要设置 set hive.exec.dynamic.partition.mode=nonstrict;

- 分区表总结:
  - 创建分区表的时候 需要通过partitioned by来指定分区字段
  - 添加分区字段的时候 可以通过alter table 表名 add if not exists partition(分区字段名='分区字段的值');
  - 如果查询的时候 可以使用分区字段，尽量利用
  - 动态分区 需要设置 set hive.exec.dynamic.partition.mode=nonstrict;


### 10_Hive的自定义函数[20:53]
- 目标: 能够使用hive自定义函数

- 内置运算符: 关系运算符, 算术运算符, 逻辑运算符, 复杂运算
- 内置函数: 
  - 简单函数: 日期函数 字符串函数 类型转换
  - 统计函数: sum avg distinct, uni
  - 集合函数
  - 分析函数
    ```sql
    show functions; 显示所有函数
    desc function 函数名;
    desc function extended 函数名;
    ```
  - 注意:
    - Hive 1.2.0之前的版本仅支持UNION ALL，其中重复的行不会被删除。 
    - Hive 1.2.0和更高版本中，UNION的默认行为是从结果中删除重复的行。
    - distinct, 只会对查询结果去重, 不会对数据去重.

- 概念:
  - UDF(user-defined function) 用户自定义函数   相当于mapper
  - UDAF(User Defined Aggregate Function) 用户自定义聚合函数:  相当于 reducer

- 使用:
  - 可以使用别人已经编译好的.jar文件作为UDF/UDAF
  - 也可以使用自己编写的python文件来作为UDF/UDAF
  - 先创建python 脚本，要处理的数据都是从sys.stdin 这里输入的
  - 把python脚本添加到Hive当中，可以从本地的linux添加，也可以放到HDFS上添加
  - 通过Transform调用UDF/UDAF
  ```python
    SELECT TRANSFORM(fname, lname) USING 'python udf.py' AS (fname, l_name) FROM u;
  ```

### 11_Hive综合案例分析[34:12]
- 需求: 根据用户行为以及文章标签筛选出用户最感兴趣(阅读最多)的标签

- 思路:
  - 准备数据
  - 把文章表中关键字列表拆开 lateral view explode
  - 根据文章id找到用户查看文章的关键字, left outer JOIN
  - 根据文章id找到用户查看文章的关键字并统计频率: group by a.user_id,b.kw
  - 将用户查看的关键字和频率合并成 key:value形式 concat_ws(':',b.kw,cast (count(1) as string))
  - 将上面聚合结果转换成map: str_to_map(concat_ws(',',collect_set(cc.kw_w))) 
  - 将用户的阅读偏好结果保存到表中: create table user_kws as 上面的查询语句
  - 从表中通过key查询map中的值:  wm['kw1'] 
  - 从表中获取map中所有的key 和 所有的value: map_keys(wm),map_values(wm) 
  - 用lateral view explode把map中的数据转换成多列:  lateral view explode(wm) t as keyword

- 注意: 
    - Could not connect to meta store using any of the URIs provided 报错信息
    - 拒接连接
    - 说明元数据存储没有打开
    



### 12_Hive综合案例完成[16:44]
- 目标: 能够根据文档实现根据用户行为以及文章标签筛选出用户最感兴趣(阅读最多)的标签. 
- 新知识点:
  - collect_set和 collect_list
  - 将group by中的某列转为一个数组返回
  - collect_list**不去重**而collect_set**去重**
  - explode
    
    - 把复杂的数据结构的内容拆开，拆成一个元素一行
  - lateral view explode
    - lateral view与explode配合 可以将explode的结果创建成一个视图 和其它列一起查询， 需要给这个侧视图起一个别名
    - ```sql
      select article_id,kw from articles lateral view outer explode(key_words) t as kw;
      ```
  - concat/concat_ws
    - 作用 字符串拼接，可以传入列名，也可以传入字符串
    - concat 会按照传入的顺序拼接字符串
    - concat_ws 传入的第一个参数 是分隔符，第二个参数可以是一个复杂的数据类型比如 array
    - 如果concat_ws传入的是array 会遍历整个数组，把每一个元素用分隔符分割拼接起来，最终得到一个字符串
  - str_to_map
    
    - 把符合 key:value形式的字符串转换成map类型
  - map类型的数据特殊的api
    - 通过map的key去查询value
      ```sql
      select user_id, wm['kw1'] from user_kws;
      ```
    - 取出所有的key 和 所有的value
      ```sql
      select user_id,map_keys(map类型字段名字),map_values(map类型字段名字) from user_kws;
      ```

- 整体代码

  ```sql
  -- 将用户的阅读偏好结果保存到表中
  create table user_kws as 
  select cc.user_id,str_to_map(concat_ws(',',collect_set(cc.kw_w))) as wm
  from(
  select a.user_id, concat_ws(':',b.kw,cast (count(1) as string)) as kw_w 
  from user_actions as a 
  left outer JOIN (select article_id,kw from articles
  lateral view outer explode(key_words) t as kw) b
  on (a.article_id = b.article_id)
  group by a.user_id,b.kw
  ) as cc 
  group by cc.user_id;
  -- 从表中通过key查询map中的值
  select user_id, wm['kw1'] from user_kws;
  -- 从表中获取map中所有的key 和 所有的value:
  select user_id,map_keys(wm),map_values(wm) from user_kws;
  -- 用lateral view explode把map中的数据转换成多列
  select user_id,keyword,weight from user_kws lateral view explode(wm) t as keyword,weight;
  ```

  

### 13_hive实现wordcount案例[05:55]

- 思路:
  ①创建一张表 表里面只有一个字段 把text.txt放上去
  ② split('字段名字','拆分的分隔符')
  ③ 拆分之后 结果变成array，需要把这个array变成 一行一个单词 一行一个单词
  ④会用到explode
  ⑤ group by 再count  可以统计单词出现的次数

- 建表
  ```sql
  CREATE EXTERNAL TABLE test1 (line string) location '/test';
  ```
- split, 用空格去split 得到一个list

  ```sql
  select split(line,' ') from test1;
  ```
- 在split基础上 去explode，把list拆开
  ```sql
  select explode(split(line,' ')) as word from test1;
  ```

- 统计结果

  ```sql
   select word, count(*) as num from (select explode(split(line,' ')) as word from test1) as t group by word;
  ```



### 14_hive内容回顾[15:9] 

##### 用户画像案例
- 协同过滤
  - 需要用到用户-物品的评分
- 基于内容的推荐
  - 用户画像
  - 物品画像
    - 建立物品画像的时候 有一部分内容是要从业务数据库中取 关系型数据-》hdfs
    - 通过 sqoop 把mysql中的数据导入到hive中（实际就是数据导入到hdfs,表的元数据通过hive去管理）
    - spark 加载hive的数据 通过spark sql做后续的处理 

##### hive内容回顾
- 本质
  - 数据仓库的工具
  - 做了两件事儿
    - 元数据的维护
    - 翻译 HQL翻译成 hdfs命令/mapreduce 作业
  - hive优点
    - 不用学mapreduce 就可以处理hdfs上的结构化数据
    - hive比直接写mapreduce代码要少很多，提高开发效率
  - hive问题
    - 只能做离线计算，受到mapreduce速度限制
  
- HIve架构
  - 用户接口 命令行
  - 元数据存储  mysql
  - HQL翻译
  
- **Hive的数据模型** ☆☆☆☆☆
  - 数据库 表 ： hdfs上的文件夹
  - 数据： 表文件夹下的文件
  
- 与传统数据库的异同
  - 数据量  HQL   离线计算  增加数据/查询数据
  - 数据类型有区别 Hive支持复杂的数据类型
  
- Hive的启动不要忘记 启动元数据服务

- HQL 常用语句和SQL区别
  - 创建表的时候 需要指定分隔符
  - 加载数据的时候使用 load data 或者直接用hadoop 命令 不建议使用Insert 
  
- Hive 内部表和外部表 ☆☆☆☆☆
  - 内部表数据是放到指定位置上 /user/hive/warehouse
  - 外部表可以在hdfs上的任意位置
  - 创建外部表的时候需要添加external关键字，并且需要通过location指定数据位置
  - 删除数据的时候 内部表会删除数据和元数据
  - 外部表只删除元数据
  
- Hive 分区表☆☆☆☆☆
  - 创建表的时候 通过partition by指定分区字段
  - 分区表实际上就是在表目录下创建子文件夹
  - 添加分区的时候通过alter table 表名 partion
  
- Hive 自定义函数
  - udf、udaf
  - 可以使用别人编写好的.jar
  - 也可以自己写python文件

- 练习hive实现wordcount 案例


### 15_作业内容说明[03:12]
- mrjob 统计 多少个单词， 多少个字符，有多少行
  ```py
  from mrjob.job import MRJob
  class MRWordCount(MRJob):
          def mapper(self, key, line):
                  yield 'lines', 1
                  yield 'chars', len(line)
                  yield 'words', len(line.split())

          def reducer(self, word, count):
                  yield word, sum(count)

  if __name__ == '__main__':
          MRWordCount.run()
  ```
  
- access.log 统计出现的ip top5
  ```python
  from mrjob.job import MRJob, MRStep
  import heapq
  class IPTop5(MRJob):

          def mapper(self, _, line):
                  rs = line.split(' ')
                  yield rs[0],1

          def reducer(self, ip, counts):
                  yield None,(sum(counts), ip)

          def top5_reducer(self, _, ip_counts):
                  for count, ip in heapq.nlargest(5, ip_counts):
                          yield ip, count

          def steps(self):
                  return [MRStep(mapper=self.mapper,reducer=self.reducer),
                                  MRStep(reducer=self.top5_reducer)]

  if __name__ == '__main__':
          IPTop5.run()
  ```



### 单节点hbase启动

- 1. 删除多slave配置, 只留master
  - `cd /root/bigdata/hbase/conf`
  - `vi regionservers`
    - 删除 hadoop-slave1 和 hadoop-slave2, 只保留master, 注释没用
    ```
    hadoop-master
    ```
  - `vi hbase-site.xml`
    删除 hadoop-slave1 和 hadoop-slave2 
    ```
      <property>
          <name>hbase.zookeeper.quorum</name>
          <value>hadoop-master:22181</value>
      </property>
    ```
- 2. 禁用防火墙
  - systemctl disable firewalld

- 3. hbase shell 运行时常见错误
  - hbase shell 进入hbase命令终端, 使用list_namespace报错
    - 错误1: org.apache.hadoop.hbase.PleaseHoldException: Master is initializing
    - 原因: 时间不一致
    - 解决: 与网络时间同步即可   
        1. 安装ntpdate工具
          ```
          yum -y install ntp ntpdate
          ```

        2. 设置系统时间与网络时间同步
          ```
          ntpdate cn.pool.ntp.org
          ```
    - 错误2: ERROR: KeeperErrorCode = NoNode for /hbase/master
      原因: 之前hbase是配置伪集群环境, 数据分散其他机子上, 导致zookeeper无法启动. 
      解决
        - 删除zk数据目录下之前生成的version-2文件夹
          ```
          cd /root/bigdata/hbase/zookeeper
          rm -rf version-2/
          ```
        - 然后重启 hbase