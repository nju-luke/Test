# -*- coding: utf-8 -*-
# @Time    : 2017/9/22
# @Author  : Luke

import os
import shutil

parent_path = "/Users/hzqb_luke/Downloads/test"

def merge():
    for secP in os.listdir(parent_path):
        if secP.startswith("."): continue
        ori_path = os.path.join(parent_path, secP)
        new_path = os.path.join(ori_path,"all_records")
        if os.path.exists(new_path):
            # os.remove(new_path)
            continue
        all_records_file = open(new_path,"w")

        for name in os.listdir(ori_path):
            if name.startswith(".") or name.startswith("all_"): continue
            print "Reading %s" % name
            file_path = os.path.join(ori_path, name)

            with open(file_path) as par_file:
                for line in par_file:
                    all_records_file.writelines(line)
        all_records_file.close()

def del_others():
    for secP in os.listdir(parent_path):
        if secP.startswith("."): continue
        ori_path = os.path.join(parent_path, secP)
        new_path = os.path.join(os.path.split(parent_path)[0],"20170923",secP)
        os.makedirs(new_path)

        for name in os.listdir(ori_path):
            if not name.startswith("all_"): continue
            file_path = os.path.join(ori_path, name)
            new_file_path = os.path.join(new_path, name)

            shutil.copy(file_path,new_file_path)

del_others()