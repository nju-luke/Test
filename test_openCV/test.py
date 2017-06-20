# -*- coding: utf-8 -*-
# @Time    : 24/11/2016 15:26
# @Author  : Luke
# @Software: PyCharm

import xlrd

file_new = open("/Users/hzqb_luke/Desktop/tmp_new.txt","w")
def foo():
    with open("/Users/hzqb_luke/Desktop/tmp.txt","r") as file_ori:
        for line in file_ori:
            items = line.split("\t")
            if len(items[0]) > 12 or "APP" in items[0]:
                line = "\t"+line
            file_new.writelines(line)
    file_new.close()

brands = [u"沃尔玛",u"好又多",u"山姆"]

def foo1():
    path_ori = "/Users/hzqb_luke/Downloads/商铺模板-1226-2.xlsx"
    workbook = xlrd.open_workbook(path_ori)
    sheet = workbook.sheet_by_name(u"模板")

    for i in range(1,434):
        row_values = sheet.row_values(i)
        if row_values[0] == u"":
            row_values[0] = tmp_city
        tmp_city = row_values[0]
        for brand in brands:
            if brand in row_values[2]:
                row_values[1] = brand
        if not row_values[0].endswith(u"市"):
            row_values[0]+=u"市"
        row_values.append(u"酒店/会场")
        row_values.append(u"高档酒店")
        file_new.writelines(("\t".join(row_values)+"\n").encode("utf-8"))
    file_new.close()



def foo2():
    path_ori = "/Users/hzqb_luke/Downloads/商铺模板-1226-2.xlsx"
    workbook = xlrd.open_workbook(path_ori)
    sheet = workbook.sheet_by_name(u"模板")

    col_values = sheet.col_values(0)
    try:
        for i in range(1,len(col_values)):
            val = col_values[i].replace(" ")
            if not val.endswith(u"市"):
                val+=u"市"
            # row_values.append(u"酒店/会场")
            # row_values.append(u"高档酒店")
            file_new.writelines((val+"\n").encode("utf-8"))#("\t".join(row_values)+"\n").encode("utf-8"))
    except IndexError:
        pass
    file_new.close()

if __name__ == '__main__':
    foo1()