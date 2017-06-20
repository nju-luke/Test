# -*- coding: utf-8 -*-
# @Time    : 24/02/2017 10:42
# @Author  : Luke
# @Software: PyCharm
import os
import shutil

path_union = "银联对应"

path_matched = "五一和银联匹配"
path_51 = "五一竖卡"

banks = [bank for bank in os.listdir(path_matched)
         if not bank.startswith(".")]

for bank in banks:
    bank_path_matched = os.path.join(path_matched,bank)
    bank_path_51 = os.path.join(path_51,bank)

    file_list_matched = [name for name in os.listdir(bank_path_matched) if not name.startswith(".")]
    file_list_51 = [name for name in os.listdir(bank_path_51) if not name.startswith(".")]

    bank_path_union = os.path.join(path_union,bank)
    if not os.path.exists(bank_path_union):
        os.mkdir(bank_path_union)

    for name in file_list_matched:
        if name in file_list_51:
            continue

        file_path_matched = os.path.join(bank_path_matched,name)
        file_path_union = os.path.join(bank_path_union,name)

        shutil.copy(file_path_matched,file_path_union)

    print bank