__author__ = 'sandip'

import gensim
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.semi_supervised import LabelPropagation
from sklearn.linear_model import LogisticRegression

def _split_text(text):
    re_exp = '[\w]+'
    word_list = re.findall(re_exp, text)
    word_list = [word for word in word_list if word not in stop_list]
    #print(word_list)
    return word_list

stop_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                            'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
                            'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
                            'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
                            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
                            'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                            'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                            'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
                            'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
                            'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
                            'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                            'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']


LabeledSentence = gensim.models.doc2vec.LabeledSentence


# Data split
whole_data = pd.read_pickle('../data_load/enron_data_new.pkl')
wharton_emails = whole_data[(whole_data['type'] == 1)]
train_wharton = wharton_emails[0:250]
test_wharton = wharton_emails[250:]

print(len(train_wharton))

non_wharton_emails = whole_data[(whole_data['type'] == 0)]
non_wharton_emails = non_wharton_emails[0:20000]
msk = np.random.rand(len(non_wharton_emails)) < 0.9
train_non_wharton = non_wharton_emails[msk]
test_non_wharton = non_wharton_emails[~msk]


imbalance = len(train_non_wharton) - len(train_wharton)
print("Imbalance:", imbalance)

train_all = train_non_wharton.append(train_wharton,ignore_index=True)
test_all = test_non_wharton.append(test_wharton, ignore_index=True)
test_all['email'] = test_all['email'].apply(lambda x: re.sub('wharton', ' ', x))


#print(test_wharton['email'].values)


train_all = train_all[train_all.email.str.len() > 20]

test_all = test_all[test_all.email.str.len() > 20]
#train_all[(train_all.type == 0), 'type'] = -1

print("data cleaned")


x_train = train_all['email'].values
x_test = test_all['email'].values
y_train = train_all['type'].values
y_test = test_all['type'].values


ix = np.where(y_train == 1)[0]



print(x_train.shape[0])


#Do some very minor text preprocessing
def cleanText(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]

    #treat punctuation as individual words
    for c in punctuation:
         corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [_split_text(z) for z in corpus]
    return corpus

x_train = np.array(cleanText(x_train))
#print(x_train)
small_values_ind = np.where(len(x_train) < 8)[0]
print(len(small_values_ind))
print(small_values_ind)
x_test = cleanText(x_test)

def labelizeReviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s' % (label_type, i)
        labelized.append(LabeledSentence(v, [label]))
        if len(v) < 6:
            print(v)
    return labelized

x_train = labelizeReviews(x_train, 'TRAIN')
x_test = labelizeReviews(x_test, 'TEST')

import random

size = 300

#instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=1, window=5, size=size, sample=1e-3,  workers=3)
#model_dbow = gensim.models.Doc2Vec(min_count=1, window=5, size=size, sample=1e-3,  dm=0, workers=3)

#build vocab over all reviews
model_dm.build_vocab(np.concatenate((x_train, x_test)))
#model_dbow.build_vocab(np.concatenate((x_train, x_test)))

print("doc2vec vocabulary built")
count = 0

for key in model_dm.vocab.keys():
    if "TRAIN" in key:
        count += 1
print("vocab size:", count)

x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
for epoch in range(20):
    perm = np.random.permutation(x_train.shape[0])
    #print(perm)
    #print(type(perm))
    model_dm.train(x_train[perm])
    #model_dbow.train(x_train[perm])

print("Doc2vec trained")

count = 0

for key in model_dm.vocab.keys():
    if "TRAIN" in key:
        count += 1
print("vocab size after train:", count)



print(model_dm.most_similar('wharton'))
#Get training set vectors from our models
def getVecs(model, corpus, size):
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

train_vecs_dm = getVecs(model_dm, x_train, size)

train_positive_dummy_vecs = []

for i in range(imbalance):
    rnd1 = np.random.choice(ix)
    rnd2 = np.random.choice(ix)
    tr_vc_1 = train_vecs_dm[rnd1]
    tr_vc_2 = train_vecs_dm[rnd2]
    new_vc = (tr_vc_1 + tr_vc_2) / 2
    train_positive_dummy_vecs.append(new_vc)

train_positive_dummy_vecs = np.asarray(train_positive_dummy_vecs)
train_positive_y = np.ones(len(train_positive_dummy_vecs))

print("Dummy vectors created")
print(len(train_positive_dummy_vecs))


#print(len(train_vecs_dm))
#print(train_vecs_dm[0])
#train_vecs_dbow = getVecs(model_dbow, x_train, size)

print(len(train_vecs_dm), train_vecs_dm.shape)
print(len(train_positive_dummy_vecs), train_positive_dummy_vecs.shape)

train_vecs = np.concatenate((train_vecs_dm,train_positive_dummy_vecs))

y_train = np.concatenate((y_train,train_positive_y))

#train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

x_test = np.array(x_test)

for epoch in range(20):
    perm = np.random.permutation(x_test.shape[0])
    model_dm.train(x_test[perm])
    #model_dbow.train(x_test[perm])

#Construct vectors for test reviews
test_vecs_dm = getVecs(model_dm, x_test, size)
#test_vecs_dbow = getVecs(model_dbow, x_test, size)

test_vecs = test_vecs_dm  #np.hstack((test_vecs_dm, test_vecs_dbow))

#lr = SVC()
#lr.fit(train_vecs, y_train)

#neigh = KNeighborsClassifier(n_neighbors=10)
#neigh.fit(train_vecs, y_train)


#label_prop_model = LabelPropagation()

logreg = LogisticRegression()
logreg.fit(train_vecs, y_train)

print("classifier trained")

pred = logreg.predict(test_vecs)
print(pred)
print(y_test)
print("precision:", metrics.precision_score(y_test, pred))
print("recall:", metrics.recall_score(y_test, pred))


labels_word_vec = list()
f_lb = open('classifier_labels.txt','r')
for line in f_lb.readlines():
    label_name = parts[2]
    parts = line.split()
    labels_word_vec.append([parts[0],parts[1],parts[2], model_GN[label_name]])

