__author__ = 'sandip'

import pandas as pd
import time
import numpy as np
import lda
import sklearn.feature_extraction.text as text

start_time = time.time()

whole_data = pd.read_pickle('enron_data.pkl')
wharton_emails = whole_data[(whole_data['type'] == 1)]
wharton_email_list = wharton_emails['email'].values

non_wharton_emails = whole_data[(whole_data['type'] == 0)]
non_wharton_emails = non_wharton_emails[0:1000]
non_wharton_email_list = non_wharton_emails['email'].values



vectorizer = text.CountVectorizer(stop_words='english', min_df=20)

dtm = vectorizer.fit_transform(wharton_email_list).toarray()
vocab = np.array(vectorizer.get_feature_names())

model = lda.LDA(n_topics=40, n_iter=500, random_state=1)
model.fit(dtm)
topic_word = model.topic_word_
n_top_words = 20

for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))