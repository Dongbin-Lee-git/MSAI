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

doc2 = open("Gettysburg.txt", "r")
doc2Txt = doc2.read()
