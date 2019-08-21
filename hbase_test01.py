
#!/c/Users/struggle6/AppData/Local/Programs/Python/Python37/python

import happybase
def getQuery():
    connection = happybase.Connection('192.168.19.137')
    # 通过connection找到user表 获得table对象
    table = connection.table('user')
    result = table.row('rowkey_22',columns=['base_info:username'])
    #result = table.row('rowkey_22',columns=['base_info:username'])
    result = table.rows(['rowkey_22','rowkey_16'],columns=['base_info:username'])
    print(result)
    # 关闭连接
    connection.close()

getQuery()



