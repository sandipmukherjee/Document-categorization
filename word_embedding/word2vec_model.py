__author__ = 'sandip'

from gensim.models.word2vec import Word2Vec
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.preprocessing import scale
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn import metrics
from sklearn.svm import SVC

whole_data = pd.read_pickle('../enron_data.pkl')

wharton_emails = whole_data[(whole_data['type'] == 1)]

non_wharton_emails = whole_data[(whole_data['type'] == 0)]
non_wharton_emails = non_wharton_emails[0:10000]

total_data = non_wharton_emails.append(wharton_emails, ignore_index=True)


x_train, x_test, y_train, y_test = train_test_split(total_data['email'].values, total_data['type'].values, test_size=0.1)

def cleanText(corpus):
    corpus = [z.lower().replace('\n','').split() for z in corpus]
    return corpus

x_train = cleanText(x_train)
x_test = cleanText(x_test)

n_dim = 300

email_w2v = Word2Vec(size=n_dim, min_count=2)
email_w2v.build_vocab(x_train)

print("vocab created")

def buildWordVector(text, size):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += email_w2v[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec

train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
train_vecs = scale(train_vecs)
email_w2v.train(x_test)

print("word vector trained")

test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
test_vecs = scale(test_vecs)

lr = SVC()
lr.fit(train_vecs, y_train)

print("svm trained")

print('Test Accuracy: %.2f' %lr.score(test_vecs, y_test))


pred = lr.predict(test_vecs)
print(pred)
print(y_test)
print("precision:", metrics.precision_score(y_test, pred))
print("recall:", metrics.recall_score(y_test, pred))
#print(email_w2v.similarity('wharton', 'farber'))