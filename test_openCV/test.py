# -*- coding: utf-8 -*-
# @Time    : 24/11/2016 15:26
# @Author  : Luke
# @Software: PyCharm

import PyPDF2


input1 = PyPDF2.PdfFileReader(open("50_平安携程卡_2344.jpg", "rb"))

input1.getPage()

print input1