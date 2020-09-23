import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pymysql
from datetime import datetime
import itertools
from multiprocessing import Pool
import multiprocessing
import numpy as np

"""
ChunkSize sub matrix for dataset;
The option is based on the RAM size
"""
ChunkSize = 5000
# Shingle length
ShingleSize = 8
# Similarity percentage
similarityPercent = 0.75
ProcessCount = multiprocessing.cpu_count()

# Create date point
start = datetime.now()


# basic text cleaning
def clean_text(text):
    text = text.split('\n')
    text = list(filter(None, text))
    text = ' '.join(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r'[^\w\s]', '', text)
    shingle = [text[i:i + ShingleSize] for i in range(len(text))][:-ShingleSize]
    return ','.join(shingle)


# creates chunks
def chunks(lst, count):
    begin = 0
    for i in range(count):
        end = begin + len(lst[i::count])
        yield lst[begin:end]
        begin = end


# MySQL connecting params
DB_Connect = pymysql.connect('localhost', 'user', 'password', 'database', charset='utf8mb4')
cursor = DB_Connect.cursor()

cursor.execute("select lower(content),id from comments where length(content) > 100 ")
w = cursor.fetchall()

# Cleaning and prepare data
corpus = [item[0] for item in w]
corpusId = [item[1] for item in w]
OriginalCorpus = corpus.copy()


def Sim(A, B, C, D):
    similarities = cosine_similarity(B[A[0]].astype(np.float32), B[A[1]].astype(np.float32))
    x, y = np.where(similarities > similarityPercent)
    res = []
    for k, j in zip(x, y):
        if D[A[0]][k] != D[A[1]][j]:
            res.append((D[A[0]][k], C[A[0]][k], similarities[k][j].item(), D[A[1]][j], C[A[1]][j]))
    return res


# optional function for duplicates remove
def Filter(List, StopList):
    if List[3] not in StopList:
        return List


if __name__ == '__main__':
    pool = Pool(processes=ProcessCount)
    data = pool.map(clean_text, corpus)

    for i in range(len(corpus)):
        corpus[i] = clean_text(corpus[i])

    count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))

    sparse_matrix = count_vectorizer.fit_transform(data)

    print('Matrix shape: {}'.format(sparse_matrix.shape))

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
    matrixCombinations = itertools.combinations_with_replacement(range(len(csrMatrix)), 2)
    s = pool.starmap(Sim, zip(matrixCombinations, itertools.repeat(csrMatrix), itertools.repeat(textArray), itertools.repeat(idArray)))

    s = [item for sublist in s for item in sublist]
    print(len(s))

    # removes duplicates. Example: for pares a = b and a = c removes pair b = c
    # stopList = [item[0] for item in s]
    # w = pool.starmap(Filter,zip(s, itertools.repeat(stopList)))
    w = s
    for i in w:
        if i is not None:
            ratio = len(i[1]) / len(i[4])
            # insert text where length difference no more than n %
            if 0.6 < ratio < 1.6:
                cursor.execute(
                    "INSERT INTO similarity (FirstId,FirstText,sim,SecondId,SecondText) VALUES (%s, %s, %s, %s, %s)",
                    (i[0], i[1], i[2], i[3], i[4]))

    DB_Connect.commit()
    DB_Connect.close()
    print(datetime.now() - start)
