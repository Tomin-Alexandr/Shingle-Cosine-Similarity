# -*- coding: utf8 -*-
import pymysql
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from multiprocessing import Pool
import concurrent.futures
from datetime import datetime
import itertools
import hashlib



DB_Connect = pymysql.connect('localhost', 'user', '25477452', 'news', charset='utf8mb4')
cursor = DB_Connect.cursor()

#cursor.execute("select lower(text),id from facebook.comments order by rand() limit 5000")
cursor.execute("select lower(title),url from news WHERE date_published  >= '2020-08-23 00:00:00' AND date_published <= '2020-09-02 00:59:59' limit 5000")
ddata = cursor.fetchall()



def chunks(lst, count):
    start = 0
    for i in range(count):
        stop = start + len(lst[i::count])
        yield lst[start:stop]
        start = stop


def Calcs(data):
    count = 0
    for couple in data:
        try:
            count += 1
            if count % 100000 == 0:
                print(count)
            text_1 = re.sub(r'[^\w\s]', '', couple[0][0])
            shingles = [text_1[i:i + 5] for i in range(len(text_1))][:-5]  # shingle size = 5
            text_2 = re.sub(r'[^\w\s]', '', couple[1][0])
            other_shingles = [text_2[i:i + 5] for i in range(len(text_2))][:-5]  # shingle size = 5


            s = len(set(shingles) & set(other_shingles)) / len(set(shingles) | set(other_shingles))
            if s > 0.65 and s != 1:
                cursor.execute("INSERT INTO ml.temp_news (url, title, rating,dublicate) VALUES (%s, %s, %s, %s)",(couple[0][1],couple[0][0],s,couple[1][0]))
                DB_Connect.commit()
        except Exception as e:
            pass



if name == 'main':
    Listfor = itertools.combinations(ddata, 2)
    Data = chunks(list(Listfor), 11)


    pool = Pool(processes=11)
    start = datetime.now()
    pool.map(Calcs, Data)

    pool.close()
    pool.join()
    Calcs(ddata)

    DB_Connect.close()

    print(datetime.now() - start)