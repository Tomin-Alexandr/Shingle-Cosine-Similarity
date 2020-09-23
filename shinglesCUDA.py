import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from datetime import datetime
import itertools
import numpy as np


# basic text cleaning
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    stemmer = SnowballStemmer("russian")
    StopWords = stopwords.words('russian') + list(string.punctuation)
    a = [i for i in word_tokenize(text.lower()) if i not in StopWords]
    a = [stemmer.stem(item) for item in a]
    return ' '.join(a)



"""
ChunkSize sub matrix for dataset;
The option is based on the RAM size
"""
ChunkSize = 10000

# Create date point
start = datetime.now()

# MySQL connecting params
DB_Connect = pymysql.connect('localhost', 'user', '25477452', 'pikabu', charset='utf8mb4')
cursor = DB_Connect.cursor()

cursor.execute("select content,id from comments where length(content)>100 limit 20000")
w = cursor.fetchall()

# Cleaning and prepare data
corpus = [item[0] for item in w]
corpusId = [item[1] for item in w]
OriginalCorpus = corpus.copy()

for i in range(len(corpus)):
    corpus[i] = clean_text(corpus[i])

count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(corpus)

print(sparse_matrix.shape)
# Create arrays for separated vectors, id and text data
csrMatrix = []
idArray = []
textArray = []
for i in range(ChunkSize, sparse_matrix.shape[0] + ChunkSize, ChunkSize):
    csrMatrix.append(sparse_matrix[i - ChunkSize:i - 1])
    idArray.append(corpusId[i - ChunkSize:i - 1])
    textArray.append(OriginalCorpus[i - ChunkSize:i - 1])

# Create combination list for matrix comparison
IterationList = itertools.product(range(len(csrMatrix)), repeat=2)

# Start to compare
for i in IterationList:
    print('Comparison {} matrix with {} matrix'.format(i[0], i[1]))
    print(i)
    # Temp similarities matrix. Uses float32 type for size reduce
    similarities = cosine_similarity(csrMatrix[i[0]].astype(np.float32), csrMatrix[i[1]].astype(np.float32))

    # Search texts where similarity more than 65 percent
    x, y = np.where(similarities > 0.65)

    # Insert data in to database
    #for k, j in zip(x, y):
        #if idArray[i[0]][k] != idArray[i[1]][j]:
            #cursor.execute("INSERT INTO similarity (FirstId,FirstText,sim,SecondId,SecondText) VALUES (%s, %s, %s, %s, %s)",
                           #(idArray[i[0]][k], textArray[i[0]][k], similarities[k][j].item(), idArray[i[1]][j], textArray[i[1]][j]))

DB_Connect.commit()
DB_Connect.close()
print(datetime.now() - start)
