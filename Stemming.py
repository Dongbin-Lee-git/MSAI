# Stemming : 단어가 같은 어근을 가지고 있는지 확인하는 기술
#            자음, 모음, 공통 어근, 문자열 조합, 문맥에 기초해 패턴을 정의 할 때 사용하는 알고리즘
from string import punctuation
from nltk.corpus import stopwords
import math
from textblob import TextBlob as tb
import nltk
import pandas as pd
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

doc4 = open("KennedyInaugural.txt", "r", encoding='latin_1')
KenTxt = doc4.read()

print(KenTxt)

# Normalize and remove stop words

KenTxt = ''.join(c for c in KenTxt if not c.isdigit())
KenTxt = ''.join(c for c in KenTxt if c not in punctuation).lower()
KenTxt = ' '.join([word for word in KenTxt.split() if word not in (stopwords.words('english'))])

# Get Frequency distribution
words = nltk.tokenize.word_tokenize(KenTxt)
fdist = FreqDist(words)
count_frame = pd.DataFrame(fdist, index=[0]).T
count_frame.columns = ['Count']

# Plot frequency
counts = count_frame.sort_values('Count', ascending=False)
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
counts['Count'][:60].plot(kind='bar', ax=ax)
ax.set_title('Frequency of the most common words')
ax.set_ylabel('Frequency of word')
ax.set_xlabel('Word')
plt.show()

from nltk.stem.porter import PorterStemmer

# Get the word stems
ps = PorterStemmer() # 단어의 형태소 분석 알고리즘
stems = [ps.stem(word) for word in words]

# Get Frequency distribution
fdist = FreqDist(stems)
count_frame = pd.DataFrame(fdist, index=[0]).T
count_frame.columns = ['Count']

# Plot frequency
counts = count_frame.sort_values('Count', ascending=False)
fig = plt.figure(figsize=(16, 9))
ax = fig.gca()
counts['Count'][:60].plot(kind='bar', ax=ax)
ax.set_title("Frequency of the most common words")
ax.set_ylabel('Frequency of word')
ax.set_xlabel('Word')
plt.show()