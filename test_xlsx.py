# -*- coding: utf-8 -*-
# @Time    : 9/30/16 17:12
# @Author  : Luke
# @Software: PyCharm

import xlrd
import xlwt
from datetime import date, datetime


def read_excel(path):
    # 打开文件
    workbook = xlrd.open_workbook(path)
    # 获取所有sheet
    print workbook.sheet_names()  # [u'sheet1', u'sheet2']
    sheet2_name = workbook.sheet_names()[1]

    # 根据sheet索引或者名称获取sheet内容
    sheet2 = workbook.sheet_by_index(1)  # sheet索引从0开始
    sheet2 = workbook.sheet_by_name('Sheet1')

    # sheet的名称，行数，列数
    print sheet2.name, sheet2.nrows, sheet2.ncols

    # 获取整行和整列的值（数组）
    rows = sheet2.row_values(3)  # 获取第四行内容
    cols = sheet2.col_values(2)  # 获取第三列内容
    print rows
    print cols

    # 获取单元格内容
    print sheet2.cell(1, 0).value.encode('utf-8')
    print sheet2.cell_value(1, 0).encode('utf-8')
    print sheet2.row(1)[0].value.encode('utf-8')

    # 获取单元格内容的数据类型
    print sheet2.cell(1, 0).ctype

def get_names(path):
    workbook = xlrd.open_workbook(path)
    sheet = workbook.sheet_by_name("Sheet1")
    card_names = sheet.col_values(3)
    figure_names = sheet.col_values(8)
    return card_names,figure_names

def build_dics(card_names,figure_names,start_num):
    name2fig = dict()
    fig2name = dict()
    for name,fig_name in zip(card_names[start_num:],figure_names[start_num:]):
        if not name in name2fig:
            name2fig[name] = fig_name
        else:
            try:
                name2fig[name].append(fig_name)
            except AttributeError:
                name2fig[name] = [name2fig[name],fig_name]
        fig2name[fig_name] = name
    return name2fig,fig2name


if __name__ == '__main__':
    path = "交行贷记卡列表.xlsx"
    read_excel(path)
    card_names, figure_names = get_names(path)
    na2fi,fi2na = build_dics(card_names,figure_names,4)
    print card_names,figure_names