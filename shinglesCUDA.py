import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from datetime import datetime
import itertools
import numpy as np


# basic text cleaning
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


# connecting stop words
STOP_WORDS = set(stopwords.words('russian'))

"""
ChunkSize sub matrix for dataset;
Based on the RAM size
"""
ChunkSize = 10000

# Create date point
start = datetime.now()

# MySQL connecting params
DB_Connect = pymysql.connect('localhost', 'user', '25477452', 'pikabu', charset='utf8mb4')
cursor = DB_Connect.cursor()

cursor.execute("select lower(content),id from comments where tag != 'Коты&Девушки' and length(content)>100")
w = cursor.fetchall()

# Cleaning and prepare data
corpus = [item[0] for item in w]
corpusId = [item[1] for item in w]
for i in range(len(corpus)):
    corpus[i] = clean_text(corpus[i])

count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(corpus)

csrMatrix = []
idArray = []
textArray = []
for i in range(ChunkSize, sparse_matrix.shape[0] + ChunkSize, ChunkSize):
    csrMatrix.append(sparse_matrix[i - ChunkSize:i - 1])
    idArray.append(corpusId[i - ChunkSize:i - 1])
    textArray.append(corpus[i - ChunkSize:i - 1])

IterationList = itertools.product(range(len(csrMatrix)), repeat=2)

for i in IterationList:
    print('Comparison {} matrix with {}'.format(i[0], i[1]))
    similarities = cosine_similarity(csrMatrix[i[0]].astype(np.float32), csrMatrix[i[1]].astype(np.float32))
    # triangle = np.triu(similarities)
    x, y = np.where(similarities > 0.65)
    for k, j in zip(x, y):
        cursor.execute("INSERT INTO similarity (FirstId,FirstText,sim,SecondId,SecondText) VALUES (%s, %s, %s, %s, %s)",
                       (idArray[i[0]][k], textArray[i[0]][k], similarities[k][j].item(), idArray[i[1]][j], textArray[i[1]][j]))

DB_Connect.commit()
DB_Connect.close()
print(datetime.now() - start)
