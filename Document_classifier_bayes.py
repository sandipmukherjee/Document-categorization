__author__ = 'sandip'

import pandas as pd
import numpy as np
from sklearn import metrics
import re
import time
import math
import pickle


class SensitivityClassifierBayes:
    def __init__(self):
        self.stop_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
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
        self.keyword_list = list()

    def _init_memory(self):
        self.sensitive_docs = list()
        self.all_docs = list()
        self.non_sensitive_docs = list()
        self.number_of_total_docs = 0
        self.number_of_positive_docs = 0
        self.number_of_negative_docs = 0
        self.word_with_target_word_count = dict()
        self.word_individual_count = dict()
        self.word_non_sensitive_count = dict()

    def _split_text(self, text):
        re_exp = '[\w]+'
        return re.findall(re_exp, text)

    def _create_dicts(self, docs):
        docs_dict = []
        for doc in docs:
            doc_dict = {}
            for word in self._split_text(doc.lower()):
                if word not in self.stop_list and len(word) >= 4:
                    doc_dict[word] = doc_dict.get(word, 0.0) + 1.0
            docs_dict.append(doc_dict)
        return docs_dict

    def _get_number_hits(self, word1):
        both_word_present = 0
        word1present = 0
        word_present_non_sensitive_docs = 0
        for email in self.all_docs:
            if word1 in email.keys():
                word1present += 1   #email.get(word1, 0.0)
        for tar_email in self.sensitive_docs:
            if word1 in tar_email.keys():
                both_word_present += 1  #tar_email.get(word1, 0.0)
        if self.number_of_negative_docs > 0:
            for non_sensitive_doc in self.non_sensitive_docs:
                if word1 in non_sensitive_doc.keys():
                    word_present_non_sensitive_docs += 1

        # up weighting because of the imbalance
        word1present = (word1present - both_word_present) + both_word_present * \
                                                            ((len(self.all_docs) / len(self.sensitive_docs)) / 3)
        both_word_present *= ((len(self.all_docs) / len(self.sensitive_docs)) / 3)
        return both_word_present, word1present

    def calculate_count_for_all_words(self):
        for email in self.all_docs:
            for word in email.keys():
                if word not in self.word_individual_count:
                    both_word_hits, only_word_hits = self._get_number_hits(word)
                    self.word_individual_count[word] = only_word_hits
                    self.word_with_target_word_count[word] = both_word_hits
                    #self.word_non_sensitive_count[word] = word_present_non_sensitive_docs
                    #print(word, only_word_hits, both_word_hits)

    def _pickle_trained_model(self):
        model = {
            'word_individual_count': self.word_individual_count,
            'word_with_target_word_count': self.word_with_target_word_count,
            'word_count_non_sensitive_doc': self.word_non_sensitive_count,
            'number_of_total_docs': self.number_of_total_docs,
            'number_of_positive_docs': self.number_of_positive_docs,
            'number_of_non_sensitive_docs': self.number_of_negative_docs,
            'keywords': self.keyword_list
        }
        pickled_model = pickle.dumps(model)
        return pickled_model

    def train(self, data_set, keywords):
        # input : list of documents
        #keyword : list of keywords
        non_sensitive_data = []
        sensitive_data = []

        for data in data_set:
            for keyword in keywords:
                if keyword in data.lower():
                    sensitive_data.append(data)
                    break
        #print(sensitive_data)
        self.keyword_list = keywords
        self._init_memory()
        self.all_docs = self._create_dicts(data_set)
        self.sensitive_docs = self._create_dicts(sensitive_data)
        #print(self.keyword_list)
        #print(self.sensitive_docs)
        self.number_of_total_docs = len(self.all_docs)
        self.number_of_positive_docs = len(self.sensitive_docs)
        self.number_of_negative_docs = len(self.non_sensitive_docs)
        self.calculate_count_for_all_words()
        return self._pickle_trained_model()

    def classify(self, test_doc, pickled_model):
        model = pickle.loads(pickled_model)
        positive_log_prob_doc = 0
        negative_log_prob_doc = 0
        word_list = self._split_text(test_doc.lower())
        for key in model['keywords']:
            if key in test_doc:
                return 1.0
        prior_sensitive = model['number_of_positive_docs'] / model['number_of_total_docs']
        prior_non_sensitive = (model['number_of_total_docs'] - model['number_of_positive_docs']) / model['number_of_total_docs']
        for word in word_list:
            if word not in self.stop_list and len(word) >= 4:
                # Add one smoothing
                #p_w = (1 + model['word_individual_count'].get(word, 0)) / len(model['word_individual_count'])
                positive_prob_word_given_class = (1 + model['word_with_target_word_count'].get(word, 0.0)) / \
                                     (model['number_of_positive_docs'] + len(model['word_individual_count']))

                negative_prob_word_given_class = (1 + model['word_individual_count'].get(word, 0.0) - model['word_with_target_word_count'].get(word, 0.0)) / \
                                         (model['number_of_total_docs'] - model['number_of_positive_docs'] +
                                          len(model['word_individual_count']))
                #print(word, model['word_individual_count'].get(word, 0.0), model['word_with_target_word_count'].get(word, 0.0), negative_prob_word_given_class)
                positive_log_prob_doc += math.log(positive_prob_word_given_class)
                negative_log_prob_doc += math.log(negative_prob_word_given_class)
                #print(word, positive_prob_word_given_class, negative_prob_word_given_class)
        doc_pos_prob = math.exp(positive_log_prob_doc + math.log(prior_sensitive))
        doc_neg_prob = math.exp(negative_log_prob_doc + math.log(prior_non_sensitive))
        #print(doc_pos_prob)
        #print(doc_neg_prob)
        if doc_pos_prob != 0:
            return doc_pos_prob / (doc_pos_prob + doc_neg_prob)
        else:
            return 0.0
