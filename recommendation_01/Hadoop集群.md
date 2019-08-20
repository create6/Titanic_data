## Hadoop

### 02_Hadoop概念及其作用[16:40]
- 目标: 知道Hadoop有什么特点

#### 1. 明确学习目标
- HDFS的使用
- HDFS的原理
- MapReduce的原理

#### 2. Hadoop概念

- 分布式的计算/分布式的存储
  - 简单的编程模型，处理集群上的大规模数据
- 可扩展
  - 万台集群
- 可靠的
  - HA(High Availability) 高可用
  - 应用层检测和处理故障，不依赖昂贵的硬件

#### 3. Hadoop能干啥

- 数据仓库
  - 保存数据的历史版本
- PB级数据的存储 处理 分析 统计等业务

### 03_Hadoop的发展历史[12: 49]
- 目标: 了解Hadoop的发展历史

- Google 三篇论文, 大数据的基石

- 分布式的计算/分布式的存储 开源框架
- 扩展性强，可靠，不需要特殊的机器
- 数据仓库 海量数据的分析存储  为机器学习深度学习提供数据和计算能力

### 04_Hadoop的组件_HDFS[07:50]
- 目标: 知道 HDFS的作用和特点
- HDFS
  - 扩展性&容错性&海量数量存储

  - 将文件切分成指定大小的数据块, 

  - 并在多台机器上保存多个副本


### 05_Hadoop的组件_MapReduce[08:28]
- 目标: 知道MapReduce如何进行分布式计算的
- MapReduce
  - 扩展性&容错性&海量数据离线处理
  - 分布式计算框架
  - **移动计算比移动数据更划算**
  - MapReduce就是分布式计算的解决方案
    - Map阶段   拆分
    - Reduce阶段   聚合


### 06_Hadoop的组件_Yarn[11:17]
- 目标: 知道Yarn的作用和Hadoop的优势

- Yarn: 另外一种资源协调者
  - Hadoop 2.X以前 没有YARN MapReduce 负责调度 也负责计算
  - Hadoop 2.X以前  Mesos 开源框架可以实现集群的资源调度和管理
  - 扩展性&容错性&多框架资源统一调度
  - 作用: 提高资源的利用率
- Hadoop优势
  - 高可靠 存储可靠 计算容错 有问题会自动调度到其它几点
  - 高扩展
  - 生态社区都比较成熟

### 07_Hadoop的启动和shell命令☆☆☆☆☆[17:21]
- 目标: 熟练使用Hadoop相关的shell命令


#### HDFS

- 防火墙:
  - 关闭防火墙: systemctl stop firewalld
  - 查看防火墙命令: systemctl status firewalld
  - 禁用防火墙自启命令: systemctl disable firewalld
  - 启动防火墙：systemctl start firewalld.service
  - 启用防火墙自启命令: systemctl enable firewalld.service
- 退出安全模式

  - hdfs dfsadmin -safemode leave
  - 或
  - hadoop dfsadmin -safemode leave


### 08_HDFS架构☆☆☆☆☆【12:55】

目标: 知道NameNode 和 DataNode的作用

- NameNode 主节点 同时只能有一个NameNode 活着
- DataNode 从节点 可以有很多个从节点
- NameNode
  - 客户端请求响应
  - 元数据的存储 meta data
    - 描述数据的数据
  - 监控DataNode的健康，如果有死掉/有问题的需要调度数据到别的节点
- DataNode
  - 负责数据的存储
  - 负责和客户端之间数据IO
  - 定期汇报自身及数据的健康情况

### 09_HDFS的安装[10:23]

目标: 了解HDFS的安装安装过程

- JDK java的开发运行环境
  - 大部分的大数据框架都是用java或者scala开发的
    - scala  Java虚拟机语言
    - java -> .class->.jar   .jar文件就是java虚拟机的可执行文件
    - scala语法和java有区别  .scala -> .class ->.jar 



* 下载 JDK 和 Hadoop, 安装 解压放到特定目录

* 配置环境

  ```python
  vi ~/.bash_profile
  
  # 配置JDK的家目录
  export JAVA_HOME=/root/bigdata/jdk
  # 配置JDK的执行文件所在路径
  export PATH=$JAVA_HOME/bin:$PATH
  # 配置HADoop的家目录
  export HADOOP_HOME=/root/bigdata/hadoop
  # 配置HADOOP执行文件所在路径
  export PATH=$HADOOP_HOME/bin:$PATH
  
  #保存退出后
  source ~/.bash_profile
  ```

* 进入到解压后的hadoop目录 修改配置文件

  * 配置文件的作用

    * hadoop-env.sh 配置hadoop环境
    * core-site.xml 指定hdfs的访问方式
    * hdfs-site.xml 指定namenode 和 datanode 的数据存储位置
    * mapred-site.xml 配置mapreduce
    * yarn-site.xml 配置yarn

  * 修改 hadoop-env.sh 

    ```sh
    #找到下面内容添加java home
    export_JAVA_HOME=/root/bigdata/jdk
    ```

  * core-site.xml

    ```xml
    <configuration>
            <property>
              		  <!--配置临时文件的配置-->
                    <name>hadoop.tmp.dir</name>
                    <value>file:/root/bigdata/hadoop/tmp</value>
            </property>
            <property>
              			<!--配置HDFS默认访问位置-->
                    <name>fs.defaultFS</name>
                    <value>hdfs://hadoop-master:9000</value>
            </property>
    </configuration>
    ```

  * hdfs-site.xml 

    ```xml
    <configuration>
            <property>
              		 <!-- 配置备份数量 -->
                    <name>dfs.replication</name>
                    <value>1</value>
            </property>
            <property>
              			<!--配置NameNode数据存储目录 -->
                    <name>dfs.name.dir</name>
                    <value>/root/bigdata/hadoop/hdfs/name</value>
            </property>
            <property>
              		  <!--配置DataNode数据存储目录 -->
                    <name>dfs.data.dir</name>
                    <value>/root/bigdata/hadoop/hdfs/data</value>
            </property>
    </configuration>
    ```

  * mapred-site.xml

    ```xml
    <configuration>
            <property>
              			<!--MapReduce框架名称  -->
                    <name>mapreduce.framework.name</name>
                    <value>yarn</value>
            </property>
            <property>
              		 <!--job-tracker交互端口   -->
                    <name>mapred.job.tracker</name>
                    <value>http://hadoop-master:9001</value>
            </property>
    </configuration>
    ```

  * yarn-site.xm

    ```xml
    <configuration>
    <!-- Site specific YARN configuration properties -->
            <property>
              		  <!-- NodeManager上运行的附属服务. 需配置成mapreduce_shuffle, 才能运行MapReduce程序-->
                    <name>yarn.nodemanager.aux-services</name>
                    <value>mapreduce_shuffle</value>
            </property>
            <property>
              			<!--ResourceManager: 主机名称(域名)-->
                    <name>yarn.resourcemanager.hostname</name>
                    <value>hadoop-master</value>
            </property>
            <property>
              		  <!-- 是否开启NodeManager的虚拟内存检查 -->
                    <name>yarn.nodemanager.vmem-check-enabled</name>
                    <value>false</value>
            </property>
    </configuration>
    ```


### 10_YARN的架构[13:10]

 目标: 知道YARN的作用和执行流程

- 资源管理者 负责集群资源的协调，通过YARN的管理，可以实现多个框架共享同一个集群

- Client: 客户端
  
- ResourceManager: 资源管理者(主)
  
  - 同一时间只能有一个活着的ResourceManager
  
- NodeManager 节点管理者(从)
  
  - 可以有多个
  
- ApplicationMaster(应用管理者)

- Container(容器, CPU, 内存等资源的一个容器, 是一个任务运行环境的抽象)

- YARN也是 master-slave结构

### 11_MapReduce介绍&MrJob实现wordcount☆☆☆

* 目标:  
  * 知道什么是MapReduce以及MapReduce编程思想
  * 能够实现MrJob实现wordcount并运行

- 什么是MapReduce
  - 分布式的计算框架
  - 可以处理大规模数据的离线计算
  - 但是不可以做实时计算延迟比较高的
  
- MapReduce既是一个计算框架也是一个编程的模型
  - MapReduce的编程思想
    - 把大量的数据拆成一份一份分别计算，再汇总结果
    - Map 拆分后的数据分别计算
    - Reduce 汇总每一部分的计算结果
  
- 安装 MRJob:  

  - pip install mrjob (我们虚拟机上移已经装好了)

- 代码

  ```python
  from mrjob.job import MRJob
  
  class MRWordCount(MRJob):
  
      #每一行从line中输入
      def mapper(self, _, line):
          for word in line.split():
              yield word,1
  
      # word相同的 会走到同一个reduce
      def reducer(self, word, counts):
          yield word, sum(counts)
  
  if __name__ == '__main__':
      MRWordCount.run()
  ```

- 运行

  - 本地运行:   python mrjobwc.py test.txt 
  - hadoop运行: python mrjobwc.py -r hadoop hdfs:///test.txt -o hdfs:///output

- 注意: 必须启动hdfs和yarn

  ````shell
  # 进入到hadoop的sbin目录下
  cd /root/bigdata/hadoop/sbin
  # 启动hdfs
  ./start-dfs.sh 
  # 启动yarn
  ./start-yarn.sh
  
  # 或者 启动所有
  ./start-all.sh
  ````

   

### 12_MRJob实现topN统计☆☆☆[15:03]
- MRJob实现MapReduce
  - hadoop 提供了一个hadoop streaming的jar包， 通过hadoop streaming 可以用python 脚本写mapreduce任务 ， hadoop streaming 做用帮助把脚本翻译成java, 使用hadoop streaming有些麻烦
    - map阶段对应一个python文件
    - reduce阶段对应一个python文件
  - MRJob用法
    - 创建一个类继承MRJob
    - 重写 mapper 和 reducer
    - 如果有多个map 和reduce 阶段 需要创建MRStep对象
    - 创建MRStep对象 可以指定每一个阶段的mapper对应的方法，reducer对应的方法，combiner对应的方法
    - 通过重写steps方法 返回MRStep的list 指定多个step的执行顺序
  
- 代码

  ```python
  import sys
  from mrjob.job import MRJob,MRStep
  import heapq
  
  class TopNWords(MRJob):
      def mapper(self, _, line):
          if line.strip() != "":
              for word in line.strip().split():
                  yield word,1
  
      #介于mapper和reducer之间，用于临时的将mapper输出的数据进行统计
      def combiner(self, word, counts):
          yield word,sum(counts)
  
      def reducer_sum(self, word, counts):
          yield None,(sum(counts),word)
  
      #利用heapq将数据进行排序，将最大的2个取出
      def top_n_reducer(self,_,word_cnts):
          for cnt,word in heapq.nlargest(2,word_cnts):
              yield word,cnt
  
      #实现steps方法用于指定自定义的mapper，comnbiner和reducer方法
      def steps(self):
          #传入两个step 定义了执行的顺序
          return [
              MRStep(mapper=self.mapper,
                     combiner=self.combiner,
                     reducer=self.reducer_sum),
              MRStep(reducer=self.top_n_reducer)
          ]
  
  def main():
      TopNWords.run()
  
  if __name__=='__main__':
      main()
  ```


### 13_MapReduce原理☆☆☆☆[08:02]

* 目标: 理解MapReduce的慢的原因

- MapReduce执行慢的原因
  - 进程模型 启动MapReduce任务启动JVM虚拟机，比较消耗时间
  - 进行数据处理的时候 没有完全在内存中处理，Map阶段处理过程中 会在磁盘上进行归并排序,和数据合并
  - Map 和Reduce之间交换数据也是在磁盘上进行的



### 14_MapReduce架构[06:12]
* 目标: 知道MapReduce 1.x 版本与 2.x版本的区别
* MapReduce架构
  * 1.x没有Yarn, JobTracker和TaskTracker进行任务调度
  * 2.x 任务调度由Yarn管理

### 15_Hadoop生态介绍[21:06]

- Hadoop

- Spark 分布式计算框架
  - spark core
  - spark sql
  - spark streaming
  - spark ml mllib
- Hive
  -  sql操作mapreduce
- HBase
  - 面向列的nosql数据库
  - 是用来保存非关系型数据的
- Zookeeper
  - 集群协调者， 帮助集群管理
  - 服务的注册于发现
  - 主节点选举
  - 数据一致性保证
- Flume
  - 日志收集系统
  - 可以监控不同的文件夹和文件，如果发现发生了变化会及时同步数据到HDFS
- Kafka
  - 消息队列
- Sqoop
  - HDFS和传统关系型数据库之间进行数据交互



### 16_HDFS的读写流程[11:05]

- 写流程
  - 客户端负责数据的拆分，拆成128MB一块的小文件
  - NameNode根据设置的副本数量，负责返回要保存的DataNode列表，如果是3副本，每一个block 返回3台DataNode的URL地址
  - DataNode 负责数据的保存，和数据的复制
    - 客户端只需要把数据和列表提交给列表中的第一台机器， DataNode之间数据复制DataNode自己完成
- 读流程
  - 客户端提交文件名给NameNode
  - NameNode返回当前文件对应哪些block,以及每一个block的所有DataNode地址
  - 客户端到地址列表中的第一台DataNode取数据

### 17_HDFS的高可用[03:36]

- 目标: 了解HDFS的高可用
- HDFS的高可用
  - 数据存储故障容错
  - 磁盘故障容错
  - DataNode故障容错
  - NameNode故障容错

### 18_Hadoop发型版本的选择[08:06]

- 企业中为了稳定性 一般都选择CDH
  - CDH 不会立刻就有最新的大数据框架版本
  - 有些部分不开源
  - 搭建: https://www.cnblogs.com/piperck/p/9944469.html
- 社区版
  - 下载最新的版本
  - 可能会存在jar包冲突的问题

### 19_大数据产品相关介绍[31:34]

- HDFS的使用 ☆☆☆☆☆
  - 启动
  - 文件的上传下载删除
- MapReduce的原理 ☆☆☆☆
  - Map
  - Reduce
- MRJob☆☆☆
  - python开发MapReduce
  - 继承MRJob  mapper  reducer
  - 练习WordCount案例
- HDFS的架构以及读写流程
  - NameNode
  - DataNode


### 20_作业
使用mrjob 统计test.txt中有 多少个单词， 多少个字符，有多少行. 
