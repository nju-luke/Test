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
startJVM(getDefaultJVMPath(), "-Djava.class.path=/home/luke/桌面/Test/ansj_seg-5.0.2-all-in-one.jar:/home/luke/桌面/Test/dic", "-Xms1g", "-Xmx1g")
DicAnalysis = JClass('org.ansj.splitWord.analysis.ToAnalysis')

Result = DicAnalysis.parse("嫩绿茶北城新馆店")

result = Result.getTerms()
for item in result:
    print item.getName(),item.getNatureStr()



shutdownJVM()