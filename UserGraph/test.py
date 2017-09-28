# -*- coding: utf-8 -*-
# @Time    : 2017/9/13
# @Author  : Luke


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = "/Users/hzqb_luke/Downloads/yangbo_result2"


records = set()
users = {}
# last_day 应该设置为当天时间
last_day = pd.datetime.now()

for name in os.listdir(path):
    file_path = os.path.join(path, name)
    with open(file_path) as curr_file:
        for line in curr_file:
            if not line.startswith("c"): continue
            if not "$" in line: continue
            user, sign_time, trade_time, price = line.strip().split("$")
            if user + trade_time in records:
                continue
            records.add(user + trade_time)

            year, month, day = map(int, sign_time.split("-"))
            sd = pd.datetime(year, month, day)
            year, month, day = map(int, trade_time.split()[0].split("-"))
            td = pd.datetime(year, month, day)

            try:
                record = users[user]
                record["date"].add(sd)
                record["date"].add(td)
                record["price"].append(price)
            except KeyError:
                record = {}
                record["date"] = {sd, td}
                record["price"] = [price]
                users[user] = record

for user in users:
    record = users[user]
    record["st"] = min(record["date"])
    record["td"] = max(record["date"])
    record["price_mean"] = np.mean(np.asfarray(record["price"]))

last_day = max([users[user]["td"] for user in users])
for user in users:
    record = users[user]
    alpha = np.exp((record["td"] - last_day).days / 365.)

    # todo 这里有问题
    if len(record["price"]) == 1:
        record["value"] = record["price_mean"] / 30 * alpha
    else:
        record["value"] = record["price_mean"] / ((record["td"] - record["st"]).days + 1) * alpha



values = [users[user]["value"] for user in users]
values = np.array(values)

sorted_values = sorted(values)
index_last = int(len(sorted_values)*0.99)
max_value = sorted_values[index_last]
values_new = sorted_values * (sorted_values < max_value) + (sorted_values > max_value) * max_value
hist = np.histogram(values_new,int(max_value))

probabilities = hist[0]/np.sum(hist[0],dtype=np.float32)
pdf = np.cumsum(probabilities)
plt.plot(hist[1][1:],1-pdf)

ids = np.argmin(abs(pdf-(1-1/np.exp(1))))
threshold = hist[1][ids]

print ids, threshold

plt.show()


