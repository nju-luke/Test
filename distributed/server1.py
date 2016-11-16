# -*- coding: utf-8 -*-
# @Time    : 10/22/16 10:04
# @Author  : Luke
# @Software: PyCharm

import tensorflow as tf

worker1 = "192.168.0.103:2222"
worker2 = "192.168.0.103:2223"
ps_hosts = ["192.168.0.103:2221"]
worker_hosts = [worker1, worker2]
cluster_spec = tf.train.ClusterSpec({ "worker": worker_hosts,"ps":ps_hosts})
server = tf.train.Server(cluster_spec, job_name="worker", task_index=0)
server.join()