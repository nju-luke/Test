# -*- coding: utf-8 -*-
# @Time    : 2017/9/11
# @Author  : Luke



import os
LTP_DATA_DIR = '/Users/hzqb_luke/anaconda/pyltp/3.3.1/ltp_data'  # ltp模型目录的路径

par_model_path = os.path.join(LTP_DATA_DIR, 'parser.model')  # 依存句法分析模型路径，模型名称为`parser.model`


from pyltp import Parser
parser = Parser() # 初始化实例
parser.load(par_model_path)  # 加载模型

words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
arcs = parser.parse(words, postags)  # 句法分析

print "\t".join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
parser.release()  # 释放模型

srl_model_path = os.path.join(LTP_DATA_DIR, 'srl')  # 语义角色标注模型目录路径，模型目录为`srl`。注意该模型路径是一个目录，而不是一个文件。

from pyltp import SementicRoleLabeller
labeller = SementicRoleLabeller() # 初始化实例
labeller.load(srl_model_path)  # 加载模型

words = ['元芳', '你', '怎么', '看']
postags = ['nh', 'r', 'r', 'v']
netags = ['S-Nh', 'O', 'O', 'O']
# arcs 使用依存句法分析的结果
roles = labeller.label(words, postags, netags, arcs)  # 语义角色标注

# 打印结果
for role in roles:
    print role.index, "".join(
        ["%s:(%d,%d)" % (arg.name, arg.range.start, arg.range.end) for arg in role.arguments])
labeller.release()  # 释放模型