# -*- coding: utf-8 -*-


# coding: utf-8

import csv

csvfile = file('1coupon_suc.csv', 'rb')
reader = csv.reader(csvfile)

lines = []
n = 1
for line in reader:
    n+=1 
    if n>500:
        break
    if len(line)>1:
        lines.append(line)
        print line
#        break

csvfile.close()

