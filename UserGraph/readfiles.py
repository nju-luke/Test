# -*- coding: utf-8 -*-
# @Time    : 2017/9/21
# @Author  : Luke

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


parent_path = "20170923"


def lines_num(path):
    """
    读取文件行数
    """
    with open(path) as cur_file:
        lines = cur_file.readlines()
    return len(lines)

def fit(xdata,ydata):
    from scipy.optimize import curve_fit
    def func(x, a, b):
        return a * np.exp(-b * x)

    plt.plot(xdata, ydata, 'b-')
    popt, pcov = curve_fit(func, xdata, ydata)
    y2 = [func(i, popt[0], popt[1]) for i in xdata]
    plt.plot(xdata, y2, 'r--')
    print popt
    plt.show()
    return popt

def records_numbers():
    """
    打印每一个二级分类的记录数量。
    """
    for name in os.listdir(parent_path):
        # if not name.endswith("_new"): continue
        if name.startswith("."): continue
        secP_path = os.path.join(parent_path,name)
        line_nums = 0
        for partition in os.listdir(secP_path):
            if not partition.startswith("part"): continue
            line_nums += lines_num(os.path.join(secP_path,partition))
        print name,line_nums

def get_users(secP):
    """
    根据二级分类代码，获得记录中对应的用户数量
    """
    secP_path = os.path.join(parent_path, str(secP)+"_new")
    users_times = {}
    users_start_date = {}
    users_last_date = {}
    for partition in os.listdir(secP_path):
        if not partition.startswith("part"): continue
        partition_path = os.path.join(secP_path,partition)
        with open(partition_path) as partition_file:
            for line in partition_file:
                try:
                    users_times[line.split()[0]] += 1
                except KeyError:
                    users_times[line.split()[0]] = 1

                date_time = line.split()[4]
                try:
                    year, month, day = map(int, date_time.split("-"))
                except:
                    continue
                date = pd.datetime(year,month,day)
                try:
                    if users_start_date[line.split()[0]] > date:
                        users_start_date[line.split()[0]] = date
                except KeyError:
                    users_start_date[line.split()[0]] = date
                try:
                    if users_last_date[line.split()[0]] < date:
                        users_last_date[line.split()[0]] = date
                except KeyError:
                    users_last_date[line.split()[0]] = date

    print len(users_times)
    times = users_times.values()
    hist = np.histogram(times,np.max(times))
    plt.plot(hist[0][:30])
    pdf = np.cumsum(hist[0][:30] / np.sum(hist[0], dtype=np.float32))
    plt.plot(pdf * np.max(times))
    print np.argmin(abs(pdf-1+np.exp(-1)))
    plt.show()

def get_users_mcc(mcc):
    partition_path = os.path.join(parent_path,mcc, "all_records_new")
    users_times = {}
    users_start_date = {}
    users_last_date = {}

    with open(partition_path) as partition_file:
        for line in partition_file:
            try:
                users_times[line.split()[0]].append(line.split()[1])
            except KeyError:
                users_times[line.split()[0]] = [line.split()[1]]

            date_time = line.split()[2]
            try:
                year, month, day = map(int,[date_time[:4],date_time[4:6],date_time[6:]])
            except:
                continue
            date = pd.datetime(year,month,day)
            try:
                if users_start_date[line.split()[0]] > date:
                    users_start_date[line.split()[0]] = date
            except KeyError:
                users_start_date[line.split()[0]] = date
            try:
                if users_last_date[line.split()[0]] < date:
                    users_last_date[line.split()[0]] = date
            except KeyError:
                users_last_date[line.split()[0]] = date
    print len(users_times)
    user_days = {}
    for user in users_start_date:
        days = (users_last_date[user]-users_start_date[user]).days
        user_days[user] = 30 if days < 30 else days
    mean_times = [np.float32(len(users_times[user]))/user_days[user] for user in users_start_date]
    mean_times = np.asfarray(mean_times) * 30
    mean_times = np.array(np.round(mean_times), np.int32)

    mean_cost = [np.mean(np.asfarray(users_times[user])) / 100. / user_days[user] for user in users_start_date]
    mean_cost = np.asfarray(mean_cost) * 30
    max_c = 5000
    hist = np.histogram(mean_cost * (mean_cost < max_c), max_c / 10)
    plt.plot(hist[0])


    hist = np.histogram(mean_times,np.max(mean_times))
    popt = fit(hist[1][1:],hist[0])

    A,B = popt
    x = hist[1][1:]/B
    y = hist[0]/A
    plt.plot(x,y)

    # 找出1/e的点
    x0 = (1+np.log(A))/B
    print x0

    # 定义指数增长的评分
    y0 = 5.
    A1 = 0.1
    B1 = np.log(y0 / A1) / x0
    X = np.arange(int(x0) * 200) / 100
    x = X[:len(X) / 2]
    y = A1 * np.exp(B1 * x)

    # 超过阈值后评分缓慢增长
    d = y0 / 5
    x1 = X[len(X) / 2:]
    y1 = (x1 - x0) * d / x0 + y0
    plt.plot(x, y)
    plt.plot(x1, y1)

    plt.show()


get_users_mcc("yangbo_5311_2900")
# get_users_mcc("yangbo_4900_2900")
