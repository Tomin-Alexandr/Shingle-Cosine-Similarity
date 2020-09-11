import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from datetime import datetime
import numpy as np
import pandas as pd


def clean_text(text):
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

STOP_WORDS = set(stopwords.words('russian'))
STOP_SYMBOLS = '.,!?:;-\n\r()/=+*"' + "'"

ChunkSize = 2000

DB_Connect = pymysql.connect('localhost', 'user', '25477452', 'pikabu', charset='utf8mb4')
cursor = DB_Connect.cursor()

cursor.execute("select lower(content),id from comments where tag != 'Коты&Девушки' and length(content)>100")
w = cursor.fetchall()
corpus = [item[0] for item in w]
corpusId = [item[1] for item in w]

start = datetime.now()
for i in range(len(corpus)):
    corpus[i] = clean_text(corpus[i])

count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(corpus)

CorpusLen = sparse_matrix.shape[0]
prev = 0

for chunk in range(ChunkSize, CorpusLen, ChunkSize):
    csrMatrix = sparse_matrix.tocsr()[prev:chunk - 1]
    temp_corpus = corpus[prev:chunk - 1]
    temp_id = corpusId[prev:chunk - 1]
    print(prev, chunk)
    doc_term_matrix = csrMatrix.todense()
    df = pd.DataFrame(doc_term_matrix, columns=count_vectorizer.get_feature_names(), index=corpus[prev:chunk - 1])
    prev = chunk

    cosMatrix = cosine_similarity(df, df)

    x, y = np.where((cosMatrix > 0.65) & (cosMatrix < 0.99))
    for k, j in zip(x, y):
        cursor.execute("INSERT INTO similarity (FirstId,FirstText,sim,SecondId,SecondText) VALUES (%s, %s, %s, %s, %s)",
                       (temp_id[k], temp_corpus[k], cosMatrix[k][j].item(), temp_id[j],temp_corpus[j]))

DB_Connect.commit()

DB_Connect.close()
print(datetime.now() - start)
