# -*- coding: utf-8 -*-
"""
Created on Sat Aug 20 16:34:29 2016

@author: luke
"""

from jpype import *
import sys
reload(sys)
sys.setdefaultencoding('utf8')

'''
在jar后面的：后指定dic所在目录即可
'''
startJVM(getDefaultJVMPath(), "-Djava.class.path=./ansj_seg-5.0.2-all-in-one.jar:dic", "-Xms1g", "-Xmx1g")
DicAnalysis = JClass('org.ansj.splitWord.analysis.ToAnalysis')

Result = DicAnalysis.parse("粥公粥婆北城新馆店嫩绿茶")

result = Result.getTerms()
for item in result:
    print item.getName(),item.getNatureStr()



shutdownJVM()