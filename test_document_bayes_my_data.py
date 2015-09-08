__author__ = 'sandip'

import codecs
import pickle
from os import listdir
from os.path import isfile, join
import re
from sensitivity_classifier_bayes import SensitivityClassifierBayes

def _split_text(text):
    re_exp = '[\w]+'
    return re.findall(re_exp, text)

#train
dir = '/home/sandip/neokami/sandipdocs/train/'
onlyfiles = [f for f in listdir('/home/sandip/neokami/sandipdocs/train/') ]
dataset = {}
for f in onlyfiles:
    with codecs.open(dir+f, "r",encoding='utf-8', errors='ignore') as myfile:
        data = myfile.read().replace('\n', '')
    dataset[f] = data

#test
dir = '/home/sandip/neokami/sandipdocs/test/'
onlyfiles = [f for f in listdir('/home/sandip/neokami/sandipdocs/test/') ]
dataset_test = {}
for f in onlyfiles:
    with codecs.open(dir+f, "r",encoding='utf-8', errors='ignore') as myfile:
        data = myfile.read().replace('\n', '')
    dataset_test[f] = data


print(len(dataset.values()))

keywords = ['credit card']

negative_emails = []

scb = SensitivityClassifierBayes()
model = scb.train(dataset.values(), keywords)
m = pickle.loads(model)
#print(m['word_with_target_word_count'])
for f in dataset_test.keys():
    classification = scb.classify(dataset_test[f], model)
    print(f, classification)