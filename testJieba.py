# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 16:34:29 2016

@author: luke
"""

import jieba.posseg as poseg
import jieba



jieba.load_userdict("userDict")
#jieba.add_word("中天MCC",100,"BRAND")
str = "陆良米线"
st = poseg.cut(str)

for s in st:
    print s.word,s.flag

print ",".join(jieba.cut(str))