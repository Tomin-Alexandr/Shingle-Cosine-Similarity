import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
import string
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from datetime import datetime
import itertools
from multiprocessing import Pool
import multiprocessing
import numpy as np


# basic text cleaning
def clean_text(text):
    try:
        text = re.sub(r"http\S+", "", text)
        text = re.sub("\n", " ", text)
        stemmer = SnowballStemmer("russian")
        StopWords = stopwords.words('russian') + list(string.punctuation)
        a = [i for i in word_tokenize(text.lower()) if i not in StopWords]
        a = [stemmer.stem(item) for item in a]
    except:
        return 're.sub is Fail'
    return ' '.join(a)


def chunks(lst, count):
    start = 0
    for i in range(count):
        stop = start + len(lst[i::count])
        yield lst[start:stop]
        start = stop


"""


"""

"""
ChunkSize sub matrix for dataset;
The option is based on the RAM size
"""
ChunkSize = 10000
ProcessCount = 10
# Create date point
start = datetime.now()

# MySQL connecting params
DB_Connect = pymysql.connect('localhost', 'user', '25477452', 'pikabu', charset='utf8mb4')
cursor = DB_Connect.cursor()

cursor.execute("select content,id from comments where length(content) > 100")
w = cursor.fetchall()

# Cleaning and prepare data
corpus = [item[0] for item in w]
corpusId = [item[1] for item in w]
OriginalCorpus = corpus.copy()


def Sim(A,B,C,D):
    similarities = cosine_similarity(B[A[0]].astype(np.float32), B[A[1]].astype(np.float32))
    # Search texts where similarity more than 65 percent
    x, y = np.where(similarities > 0.65)

    for k, j in zip(x, y):
        if D[A[0]][k] != D[A[1]][j]:
            cursor.execute(
                "INSERT INTO similarity (FirstId,FirstText,sim,SecondId,SecondText) VALUES (%s, %s, %s, %s, %s)",
                (D[A[0]][k],C[A[0]][k], similarities[k][j].item(), D[A[1]][j],C[A[1]][j]))
    DB_Connect.commit()
    return 0


if __name__ == '__main__':

    pool = Pool(processes=ProcessCount)

    data = pool.map(clean_text, corpus)

    # for i in range(len(corpus)):
    #    corpus[i] = clean_text(corpus[i])

    count_vectorizer = CountVectorizer()
    sparse_matrix = count_vectorizer.fit_transform(data)

    print(sparse_matrix.shape)
    # Create arrays for separated vectors, id and text data
    csrMatrix = []
    idArray = []
    textArray = []
    for i in range(ChunkSize, sparse_matrix.shape[0] + ChunkSize, ChunkSize):
        temp = sparse_matrix[i - ChunkSize:i - 1]
        idArray.append(corpusId[i - ChunkSize:i - 1])
        textArray.append(OriginalCorpus[i - ChunkSize:i - 1])


        csrMatrix.append(temp)

    # Start to compare
    qw = itertools.product(range(len(csrMatrix)), repeat=2)


    s = pool.starmap(Sim,zip(qw, itertools.repeat(csrMatrix), itertools.repeat(textArray),itertools.repeat(idArray)))

    # Temp similarities matrix. Uses float32 type for size reduce

    DB_Connect.commit()
    DB_Connect.close()
    print(datetime.now() - start)

