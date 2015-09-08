__author__ = 'sandip'

import numpy
import logging
import os
import gensim
import wordcloud

NUM_TOPICS = 2

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

dictionary = gensim.corpora.Dictionary.load("whsamples.dict")
corpus = gensim.corpora.MmCorpus("whsamples.mm")
scalar = 0.00
eta = numpy.ones((NUM_TOPICS, len(dictionary))) * scalar
wharton = dictionary.token2id[u'wharton']
eta[0, wharton] *= 1000
# Project to LDA space
lda = gensim.models.LdaModel(corpus, id2word=dictionary, num_topics=NUM_TOPICS, passes=20, eta=eta)
final_topics = lda.print_topics(NUM_TOPICS)
logfile = open('lda_topics.txt', 'w')
print(lda.print_topics(NUM_TOPICS), end="", file=logfile)

