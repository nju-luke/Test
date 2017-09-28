# -*- coding: utf-8 -*-
# @Time    : 2017/9/26
# @Author  : Luke

import os
import matplotlib.pyplot as plt
import pandas  as pd
import numpy as np
from scipy.optimize import curve_fit


def read_file(file_path):
    """
    读取记录文件，至少包含三列：用户，金额，时间()
    :param file_path:
    :return:
    records：字典，元素为：{用户:[金额，金额，金额]}
    user_days: 用户从发起第一笔交易至最后一笔交易的间隔时间，以天为单位，最小30
    """
    records = {}
    users_start_date = {}
    users_last_date = {}

    with open(file_path) as partition_file:
        for line in partition_file:
            try:
                records[line.split()[0]].append(np.float32(line.split()[1])/100)
            except KeyError:
                records[line.split()[0]] = [np.float32(line.split()[1])/100]

            date_time = line.split()[2]
            try:
                year, month, day = map(int, [date_time[:4], date_time[4:6], date_time[6:]])
            except:
                continue
            date = pd.datetime(year, month, day)
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
    user_days = {}
    for user in users_start_date:
        days = (users_last_date[user] - users_start_date[user]).days
        user_days[user] = 30 if days < 30 else days

    return records, user_days


class ScoringOfCostAndFrequency():
    def __init__(self, records, user_days):
        self.set_parameters()
        self.build_model_frequency(records, user_days)
        self.build_model_cost(records, user_days)


    def set_parameters(self):
        """
        设定模型参数，其中
        a1，y0: 指数模型的参数
        d: 线性模型参数，决定了线性模型的斜率(self.d/x0)
        """
        self.a1 = 0.1
        self.y0 = 5.
        self.d = self.y0/5
        self.belta1 = 2.    #频次所占权重

    def data_select(self,data_hist):
        """
        筛选99%的数据，其余的不做考虑
        """
        index = np.argmin(abs(np.cumsum(data_hist,dtype=np.float32)/np.sum(data_hist) - 0.99))
        data = data_hist[:index+1]
        return data,index

    def func(self, x, a, b):
        return a * np.exp(-b * x)

    #指数拟合消费频率、平均月消费金额。（这一步的目的在于将所有用户的交易进行统计，综合评分）
    def expotional_fit(self,xdata,ydata):
        # plt.plot(xdata, ydata, 'b-')
        popt, pcov = curve_fit(self.func, xdata, ydata)
        # y2 = [self.func(x, popt[0], popt[1]) for x in xdata]
        # plt.plot(xdata, y2, 'r--')
        # print popt
        # plt.show()
        x0 = -np.log(0.1)/popt[1] + 1
        return x0

    # 月均频次模型
    def build_model_frequency(self, records, user_days):
        mean_frequency = [np.float32(len(records[user])) / user_days[user] for user in records]
        mean_frequency = np.asfarray(mean_frequency) * 30
        mean_frequency = np.array(np.round(mean_frequency), np.int32)
        hist_frequency = np.histogram(mean_frequency, np.max(mean_frequency))
        self.frequency = self.expotional_fit(hist_frequency[1][1:],hist_frequency[0])
        self.b1_f = np.log(self.y0/self.a1)/self.frequency

    def predict_frequency(self,frequency):
        if frequency < self.frequency:
            return self.a1*np.exp(self.b1_f*frequency)
        else:
            return self.d*(frequency-self.frequency)/self.frequency + self.y0

    # 月均消费模型
    def build_model_cost(self, records, user_days):
        mean_cost = [np.mean(np.asfarray(records[user])) / 100. / user_days[user] for user in records]
        mean_cost = np.asfarray(mean_cost) * 30
        hist_cost_ori = np.histogram(mean_cost,int(np.max(mean_cost)))
        counts,index = self.data_select(hist_cost_ori[0])
        costs = hist_cost_ori[1][1:index+2]
        # hist_cost_new = np.histogram(hist_cost_ori,int(counts[index]/10))
        # self.cost = self.expotional_fit(hist_cost_new[1][1:],hist_cost_new[0])
        self.cost = self.expotional_fit(counts,costs)
        self.b1_c = np.log(self.y0/self.a1)/self.cost

    def predict_cost(self,cost):
        if cost < self.cost:
            return self.a1*np.exp(self.b1_c*cost)
        else:
            return self.d*(cost-self.cost)/self.cost + self.y0

    def predict(self,user_data):
        frequency = user_data[0]
        v1 = self.predict_frequency(frequency)

        cost = user_data[1]
        v2 = self.predict_cost(cost)

        v = v1*v2/(v1+self.belta1*v2)
        return v


if __name__ == '__main__':
    trade_file_path = "20170923/yangbo_5411_2900/all_records_new"
    records, user_days = read_file(trade_file_path)
    scoring = ScoringOfCostAndFrequency(records, user_days)
    print scoring.predict([10,200])
    print scoring.predict([5,200])
    print scoring.predict([10,100])
    print scoring.predict([5,100])
    print scoring.predict([2,10000])



