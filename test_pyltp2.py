# -*- coding: utf-8 -*-
# @Time    : 2017/9/11
# @Author  : Luke



import os
LTP_DATA_DIR = '/Users/hzqb_luke/Documents/auxiliary/ltp_data'  # ltp模型目录的路径


name = '小胖子龙虾馆建设路店，地址：邯山区浴新南大街178号浴新南大街与农林路交叉口东北角'
addr = "邯山区浴新南大街178号浴新南大街与农林路交叉口东北角"

cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')  # 分词模型路径，模型名称为`cws.model`
from pyltp import Segmentor
segmentor = Segmentor()
segmentor.load(cws_model_path)
words = segmentor.segment(name)
print "\t".join(words)

pos_model_path = os.path.join(LTP_DATA_DIR,'pos.model')
from pyltp import Postagger
postagger = Postagger()
postagger.load(pos_model_path)
words = [word for word in words]
postags = postagger.postag(words)
postags = [postag for postag in postags]
print "\t".join(postags)


ner_model_path = os.path.join(LTP_DATA_DIR,'ner.model')
from pyltp import NamedEntityRecognizer
recognizer = NamedEntityRecognizer()
recognizer.load(ner_model_path)
nertags = recognizer.recognize(words,postags)
nertags = [nertag for nertag in nertags]
print "\t".join(nertags)


par_model_path=os.path.join(LTP_DATA_DIR,"parser.model")
from pyltp import Parser
parser= Parser()
parser.load(par_model_path)
arcs = parser.parse(words,postags)
arcs = ["%d:%s" % (arc.head, arc.relation) for arc in arcs]
print "\t".join(arcs)

segmentor.release()
postagger.release()
recognizer.release()
parser.release()

print "Done"

