__author__ = 'sandip'

import gensim
import time
from gensim import corpora, models, similarities
import pandas as pd
import numpy as np
import logging
import os
import nltk
import gensim


models.ldamodel.np = models.ldamodel.numpy

def iter_docs(stoplist):

    whole_data = pd.read_pickle('enron_data.pkl')
    wharton_emails = whole_data[(whole_data['type'] == 1)]
    train_wharton = wharton_emails[0:200]

    non_wharton_emails = whole_data[(whole_data['type'] == 0)]
    non_wharton_emails = non_wharton_emails[0:10000]
    msk = np.random.rand(len(non_wharton_emails)) < 0.9
    train_non_wharton = non_wharton_emails[msk]
    test_non_wharton = non_wharton_emails[~msk]

    train_all = train_non_wharton.append(train_wharton, ignore_index=True)
    total_email_list = train_all['email'].values
    for text in total_email_list:
        yield (x for x in
            gensim.utils.tokenize(text, lowercase=True, deacc=True, errors="ignore")
                                                                if x not in stoplist and len(x) > 2)

class MyCorpus(object):

    def __init__(self, stoplist):
        self.stoplist = stoplist
        self.dictionary = gensim.corpora.Dictionary(iter_docs(stoplist))

    def __iter__(self):
        for tokens in iter_docs(self.stoplist):
            yield self.dictionary.doc2bow(tokens)


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

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
stoplist = set(stop_list)
corpus = MyCorpus(stoplist)

corpus.dictionary.save("whsamples.dict")
gensim.corpora.MmCorpus.serialize("whsamples.mm", corpus)























#
# start_time = time.time()
#
# whole_data = pd.read_pickle('enron_data.pkl')
# wharton_emails = whole_data[(whole_data['type'] == 1)]
# wharton_email_list = wharton_emails['email'].values
# #print(wharton_email_list)
#
# stoplist = set('for a of the and to in'.split())
# texts = [[word for word in document.lower().split() if word not in stoplist]
#        for document in wharton_email_list]
#
# from collections import defaultdict
# frequency = defaultdict(int)
# for text in texts:
#     for token in text:
#         frequency[token] += 1
#
# texts = [[token for token in text if frequency[token] > 1]
#           for text in texts]
#
# from pprint import pprint
# dictionary = corpora.Dictionary(texts)
# corpus = [dictionary.doc2bow(text) for text in texts]
# model = models.ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=100)
# print(model.top_topics(corpus, num_words=20))
# #print(model.print_topics(num_topics=10, num_words=10))

