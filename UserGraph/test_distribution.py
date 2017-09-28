# -*- coding: utf-8 -*-
# @Time    : 10/13/16 09:51
# @Author  : Luke
# @Software: PyCharm

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "/Users/hzqb_luke/Downloads/yangbo_result1"

users = []
prices = []
trade_times = []

user_trade_time = set()

for name in os.listdir(path):
    file_path = os.path.join(path,name)
    with open(file_path) as curr_file:
        for line in curr_file:
            if not line.startswith("c"): continue
            if not "$" in line: continue
            user, sign_time, trade_time, price = line.strip().split("$")
            if user+trade_time in user_trade_time:
                continue
            user_trade_time.add(user+trade_time)
            trade_times.append(trade_time)
            users.append(user)
            prices.append(price)

year,month,day = map(int,sign_time.split("-"))
sd = pd.datetime(year,month,day)
year=int(trade_time[:4])
month = int(trade_time[4:6])
day=int(trade_time[6:])
td = pd.datetime(year,month,day)
d = td-sd
d.days
