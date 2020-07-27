doc1 = open("Moon.txt", "r")

doc1Txt = doc1.read()

# Normalize the Text
from string import punctuation

# remove numeric digits(숫자제거)
txt = ''.join(c for c in doc1Txt if not c.isdigit())
# remove punctuation and make Lower case(기호제거, 소문자로 변환)
txt = ''.join(c for c in txt if c not in punctuation).lower()

import nltk
import pandas as pd
from nltk.probability import FreqDist
nltk.download("punkt")

words = nltk.tokenize.word_tokenize(txt)
fdist = FreqDist(words)
count_frame = pd.DataFrame(fdist, index=[0]).T
count_frame.columns = ['Count']
print(count_frame)

import matplotlib.pyplot as plt
counts = count_frame.sort_values('Count', ascending=False)
fig = plt.figure(figsize=(16,9))
ax = fig.gca()
counts['Count'][:60].plot(kind='bar', ax=ax)
ax.set_title('Frequency of the most common words')
ax.set_ylabel('Frequency of word')
ax.set_xlabel('Word')
plt.show()

nltk.download("stopwords")
from nltk.corpus import stopwords
txt = ' '.join([word for word in txt.split() if word not in (stopwords.words('english'))])

words = nltk.tokenize.word_tokenize(txt)
fdist = FreqDist(words)
count_frame = pd.DataFrame(fdist, index =[0]).T
count_frame.columns = ['Count']

counts = count_frame.sort_values('Count', ascending=False)
fig = plt.figure(figsize=(16,9))
ax = fig.gca()
counts['Count'][:60].plot(kind='bar', ax = ax)
ax.set_title('Frequency of the most common words')
ax.set_ylabel('Frequency of word')
ax.set_xlabel('Word')
plt.show()

# ------------------------------------------
# TF(Term Frequency)/IDF(Inverse Document Frequency)
print("-------------------------------------------------------")

doc2 = open("Gettysburg.txt", "r")
doc2Txt = doc2.read()
print(doc2Txt)
txt2 = ''.join(c for c in doc2Txt if not c.isdigit())
txt2 = ''.join(c for c in txt2 if c not in punctuation).lower()
txt2 = ' '.join([word for word in txt2.split() if word not in (stopwords.words('english'))])

# and a third
print("-------------------------------------------------------")
doc3 = open("Cognitive.txt", "r")
doc3Txt = doc3.read()
print(doc3Txt)
txt3 = ''.join(c for c in doc3Txt if not c.isdigit())
txt3 = ''.join(c for c in txt3 if c not in punctuation).lower()
txt3 = ' '.join([word for word in txt3.split() if word not in (stopwords.words('english'))])

# define functions for TF-IDF
import math
from textblob import TextBlob as tb

def tf(word, doc): # 문서에서 빈도계산 함수
    return doc.words.count(word) / len(doc.words)

def contains(word, docs): # 단어가 문서에서 포함이 되어있는지
    return sum(1 for doc in docs if word in doc.words)

def idf(word, docs): # 역문서 빈도 계산
    return math.log(len(docs) / (1 + contains(word, docs)))

def tfidf(word, doc, docs):
    return tf(word, doc) *  idf(word, docs)

doc1 = tb(txt)
doc2 = tb(txt2)
doc3 = tb(txt3)
docs = [doc1, doc2, doc3]

# Use TF-IDF to get the three most important words frm each document
print('----------------------------------------------------')
for i, doc in enumerate(docs):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, doc, docs) for word in doc.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))
