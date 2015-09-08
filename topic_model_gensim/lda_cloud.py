__author__ = 'sandip'

import os
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

MODELS_DIR = "models"

final_topics = open("lda_topics.txt", 'r', encoding='ascii')
curr_topic = 0
lines = ""
for line in final_topics:
    lines = line
lines = lines.strip("[").strip("]")
for line in lines.split(","):
    scores = [float(x.split("*")[0].replace("'","")) for x in line.split(" + ")]
    words = [x.split("*")[1] for x in line.split(" + ")]
    freqs = []
    for word, score in zip(words, scores):
        freqs.append((word, score))
    wc = WordCloud(background_color="white", max_words=2000,
               stopwords=STOPWORDS.add("said"))
    elements = wc.fit_words(frequencies=freqs)
    plt.imshow(elements)
    plt.axis("off")
    plt.show()
    curr_topic += 1
final_topics.close()
