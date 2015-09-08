__author__ = 'sandip'


from gensim.models import Phrases
import gensim
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
import numpy as np
import re
import pandas as pd
import time

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



creed = """This is my rifle. There are many like it, but this one is mine.
My rifle is my best friend. It is my super life. I must master it as I must master my super life.
My rifle, without me, is useless. Without my rifle, I am useless. I must fire my rifle true. I must shoot straighter than my enemy who is trying to kill me. I must shoot him before he shoots me. I will...
My rifle and I know that what counts in war is not the rounds we fire, the noise of our burst, nor the smoke we make. We know that it is the hits that count. We will hit...
My rifle is human, even as I, because it is my super life. Thus, I will learn it as a brother. I will learn its weaknesses, its strength, its parts, its accessories, its sights and its barrel. I will keep my rifle clean and ready, even as I am clean and ready. We will become part of each other. We will...
Before God, I swear this creed. My rifle and I are the defenders of my country. We are the masters of our enemy. We are the saviors of my super life.
So be it, until victory is America's and there is no enemy, but peace!
"""
start_time = time.time()

whole_data = pd.read_pickle('../data_load/enron_data_new.pkl')
wharton_emails = whole_data[(whole_data['type'] == 1)]
train_wharton = wharton_emails[0:250]
test_wharton = wharton_emails[250:]
train_wharton = train_wharton.append(train_wharton, ignore_index=True)
train_wharton = train_wharton.append(train_wharton, ignore_index=True)

print(train_wharton)

non_wharton_emails = whole_data[(whole_data['type'] == 0)]
non_wharton_emails = non_wharton_emails[0:50000]
msk = np.random.rand(len(non_wharton_emails)) < 0.9
train_non_wharton = non_wharton_emails[msk]
test_non_wharton = non_wharton_emails[~msk]

train_all = train_non_wharton.append(train_wharton,ignore_index=True)
test_all = test_non_wharton.append(test_wharton, ignore_index=True)
#test_all['email'] = test_all['email'].apply(lambda x: re.sub('wharton', ' ', x))

# Convert to list of sentences, which are lists of words
#sentences = [word_tokenize(sent) for sent in train_all['email'].values.tolist()]

sentence = train_all['email'].values.tolist()
#sentence = creed.split('\n')
tokenizer = RegexpTokenizer(r'\w+')
#tokens = tokenizer.tokenize(sentence)

filtered_words = []
sentences = [tokenizer.tokenize(sent) for sent in sentence]
for tokens in sentences:
    filtered_words.append([w for w in tokens if not w in stop_list])

p0 = Phrases(filtered_words, min_count=1, threshold=10)
p = Phrases(p0[filtered_words], min_count=1, threshold=10)

#test = 'i love my rifle, i take it everywhere, all my super life'
test_list = test_wharton['email'].values.tolist()
for i in range(0, 90):
    test = test_list[i]
    test_tokens = tokenizer.tokenize(test)
    test_filtered_words = [w for w in test_tokens if not w in stop_list]
    print(p[test_filtered_words])

print("Time spent:", time.time()-start_time, " seconds")

