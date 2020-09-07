import re
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from datetime import datetime
import numpy as np
import pandas as pd

STOP_WORDS = set(stopwords.words('russian'))
STOP_SYMBOLS = '.,!?:;-\n\r()/=+*"' + "'"


def delete_bad_string(text):
    text = re.sub('[\f\t(\uf101)●➔·•]', '', text)  # delete wrong symbols
    text = re.sub('\n{2,}', '\n', text)  # repeat \n
    text = re.sub('( +)\n', '', text)  # empty string
    text = re.sub(',{2,}', ',', text)  # repeat ,
    text = re.sub(' {2,}', ' ', text)  # repeat space
    text = re.sub('-{2,}', '-', text)  # repeat -
    a = text.split('\n')
    a = [item.strip() for item in a if len(item) > 10]  # delete string if len is less then 10 symbols
    a = [item for item in a if not re.search('^([Ff]igure|[Tt]able)', item)]  # strings without Figure and Table
    a = [item for item in a if not re.search('([.…]( ?)(\d+)$)', item)]  # delete content
    a = [item for item in a if not re.search('\d', item)]
    return ' '.join(a)


def canonize_words(source):
    return [x for x in [re.sub(r'[\.,!?:\-;\n\r()/=\+\*&#%$@\'\"●•·➔]', '', y) for y in source.lower().split()] if
            x and (x not in STOP_WORDS)]


def clean_text(text):
    # return canonize_words(delete_bad_string(text))
    return delete_bad_string(text)


DB_Connect = pymysql.connect('localhost', 'user', '25477452', 'pikabu', charset='utf8mb4')
cursor = DB_Connect.cursor()

# cursor.execute("select lower(text),id from facebook.comments order by rand() limit 5000")
cursor.execute("select lower(content),id from comments order by id limit 55000")
Data = cursor.fetchall()
corpus = [item[0] for item in Data]
corpusId = [item[1] for item in Data]

start = datetime.now()
for i in range(len(corpus)):
    corpus[i] = clean_text(corpus[i])

count_vectorizer = CountVectorizer()

sparse_matrix = count_vectorizer.fit_transform(corpus)
doc_term_matrix = sparse_matrix.todense()

df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names(), index=corpus)

prev = 0
count = 0
for i in range(500, len(df.index), 500):
    temp_corpus = corpus[prev:i]
    temp_id = corpusId[prev:i]
    cosMatrix = cosine_similarity(df[prev:i], df[prev:i])
    prev = i
    x, y = np.where((cosMatrix > 0.65) & (cosMatrix < 0.99))

    for k, j in zip(x, y):
        cursor.execute(
            "INSERT INTO similarity (FirstId, FirstText, sim, SecondId, SecondText) VALUES (%s, %s, %s, %s, %s)",
            (temp_id[k], temp_corpus[k], cosMatrix[k][j], temp_id[j], temp_corpus[j]))

    count += 1
    if count % 100 == 0:
        print(count)
print(datetime.now() - start)
