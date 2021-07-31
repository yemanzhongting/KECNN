import requests
import pymysql,random,datetime
import re,time,datetime

def create_table(database):
    db = pymysql.connect(host='localhost', port=3306, user='root', password='mysql', db='water',
                          charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    cursor = db.cursor()

    sql= "DROP TABLE IF EXISTS " + str(database)

    cursor.execute(sql)

    sql2="""
        CREATE
    TABLE
    {database}
    (
    station VARCHAR(40),
    v float,
    avgv float,
    c_time date,
    c_hour VARCHAR(5),
    c_type VARCHAR(10),
    other_field VARCHAR(40)
    )ENGINE=innodb DEFAULT CHARSET=utf8;
    """
    cursor.execute(sql2.format(database=database))
    db.close()
#station,v,avgv,c_time,c_hour,c_type,other_field
def insertIntoChannel(station,v,avgv,c_time,c_hour,c_type,other_field,cursor,db):

        list=[station,v,avgv,c_time,c_hour,c_type,other_field]
        print(list)
        cursor.execute("insert into Reservoir(station,v,avgv,c_time,c_hour,c_type,other_field) \
                      values('%s','%f','%f','%s','%s','%s','%s')" % \
                       (station,v,avgv,c_time,c_hour,c_type,other_field))
        db.commit()

def get_time(timeStamp):
    timeStamp = int(timeStamp/1000)
    timeArray = time.localtime(timeStamp)
    otherStyleTime = time.strftime("%Y-%m-%d %H:%M:%S",timeArray)
    return otherStyleTime

import datetime
def get_time_list(begin,end):

    d = begin
    time_List = []
    delta = datetime.timedelta(days=1)
    while d <= end:
        # print (d.strftime("%Y-%m-%d"))
        time_List.append(d.strftime("%Y-%m-%d"))
        d += delta
    return (time_List)
# while(1):
#
#     urls=['http://www.cjh.com.cn/sqindex.html','http://zy.cjh.com.cn/sqall.html']
#     for url in urls:
#         ###开启数据库
#         db = pymysql.connect(host='localhost', port=3306, user='root', password='mysql', db='water',
#                                      charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
#         cursor = db.cursor()
#
#         content=requests.get(url)
#
#         tmp=re.findall( 'var sssq = (.*?)]',content.text)
#         data=eval(tmp[0]+"]")
#         crawl_date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         print(data)
#         for i in data:
#             print(i)
#             insertIntoChannel(i['stnm'],float(i['z']),crawl_date,get_time(i['tm']),float(i['q']),i['rvnm'].strip(),cursor,db)
#             print('插入一条')
#
#         #关闭数据库
#         cursor.close()
#     time.sleep(3600)

from jsonpath import jsonpath
import json

if __name__=='__main__':

    # 会执行删表操作，千万注意使用  三峡水库
    # create_table('Reservoir')

    # date_list=get_time_list(datetime.date(2020, 1, 11),datetime.date(2020, 8, 18))

    date_list = get_time_list(datetime.date(2000, 1, 1), datetime.date(2020, 1, 10))

    url_list={
        '三峡':'https://www.ctg.com.cn/eportal/ui?moduleId=50c13b5c83554779aad47d71c1d1d8d8&&struts.portlet.mode=view&struts.portlet.action=/portlet/waterFront!getDatas.action',
        '葛洲坝':'https://www.ctg.com.cn/eportal/ui?moduleId=622108b56feb41b5a9d1aa358c52c236&&struts.portlet.mode=view&struts.portlet.action=/portlet/waterFront!getDatas.action',
        '向家坝':'https://www.ctg.com.cn/eportal/ui?moduleId=3245f9208c304cfb99feb5a66e8a3e45&&struts.portlet.mode=view&struts.portlet.action=/portlet/waterFront!getDatas.action',
        '洛溪渡':'https://www.ctg.com.cn/eportal/ui?moduleId=8a2bf7cbd37c4d4f961ed1a6fbdf1ea8&&struts.portlet.mode=view&struts.portlet.action=/portlet/waterFront!getDatas.action',
    }

    crawl_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for i in date_list:
        data={
            'time':i
        }
        for key, value in url_list.items():
            ###开启数据库
            db = pymysql.connect(host='localhost', port=3306, user='root', password='mysql', db='water',
                                 charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
            cursor = db.cursor()

            content = requests.post(value,data=data).text

            data = json.loads(content)

            for j in ['$..ckList','$..rkList','$..xyList','$..syList']:
                try:
                    WORK_NAME = jsonpath(data,j)
                    for k in WORK_NAME[0]:
                        insertIntoChannel(key, float(k['v']), float(k['avgv']), i, k['time'], j[-6:],k['senId'],cursor, db)
                        # station, v, avgv, c_time, c_hour, c_type, other_field
                        print('插入一条')
                except:
                    pass
            # 关闭数据库
            cursor.close()
            #入库出库 上游下游