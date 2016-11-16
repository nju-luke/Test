# -*- coding: utf-8 -*-
# @Time    : 16-8-31 上午11:29
# @Author  : Luke
# @Software: PyCharm


first = True
path  = "datasetSentences.txt"
newFile = open("data.txt",'w')
with open(path,'r') as file:
    for line in file:
        if first:
            first = False
            continue
        lineList = line.strip().split()
        newLine = " ".join(lineList[1:])
        newFile.writelines(newLine+"\n")
newFile.close()
