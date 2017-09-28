# -*- coding: utf-8 -*-
# @Time    : 2017/9/25
# @Author  : Luke

import numpy as np
import os
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

def mcc_drop(users):
    mccs_times = {}
    for mccs in users.values():
        for mcc in mccs:
            try:
                mccs_times[mcc] += 1
            except KeyError:
                mccs_times[mcc] = 1
    for mcc in mccs_times:
        if mccs_times[mcc] < 2:
            mccs_times.pop(mcc)
    for user in users:
        for i in range(len(users[user]))[::-1]:
            if not users[user][i] in mccs_times:
                users[user].pop(i)

users = {}
with open("/Users/hzqb_luke/Downloads/20170923/yangbo_all_records_1000/all_records") as record_file:
    for line in record_file:
        line = line.replace("\\N", "")
        items = line.strip().split("\t")
        if len(items) < 3 or len(items[3]) != 15:
            print line
            continue
        if items[3][7:11].startswith("0"):
            continue
        try:
            users[items[0]].append(items[3][7:11])
        except KeyError:
            users[items[0]] = [items[3][7:11]]

from scipy import sparse

corpus = [" " .join(user) for user in users.values()]
vectorizer = CountVectorizer()
transformer = TfidfTransformer()
vectors = vectorizer.fit_transform(corpus)
tfidf = transformer.fit_transform(vectors)

mccs = vectorizer.get_feature_names()
weight = tfidf.toarray()
for i in range(len(weight)):
    print weight[i]