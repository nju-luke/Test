# -*- coding: utf-8 -*-
# @Time    : 16-8-31 下午2:57
# @Author  : Luke
# @Software: PyCharm



n = 0
'''
save.py

import tensorflow as tf

l = 'abcdefg'
vocabulary = [cha for cha in l]
index = range(len(vocabulary))

aList = range(10)
with tf.variable_scope("embeedings"):
    embeedings = tf.Variable(
        tf.random_uniform([len(vocabulary),5],-1,1),name="embeedings")
    a = tf.Variable(aList,dtype=tf.float32,name = "a")
embed = tf.nn.embedding_lookup(embeedings,index)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
embed = sess.run(embed)

saver = tf.train.Saver([embeedings,a])
saver.save(sess,'embeedings')

print sess.run(a)

print embed

'''



'''
restore.py

l = 'abcdefg'
vocabulary = [cha for cha in l]
index = range(len(vocabulary))
np.random.shuffle(index)

with tf.variable_scope("embeedings"):
    embeedings = tf.Variable(
        tf.random_uniform([len(vocabulary),5],-1,1),name="embeedings")
    a = tf.Variable([0]*10, dtype=tf.float32, name="a")

sess = tf.Session()

saver = tf.train.Saver([embeedings,a])
saver.restore(sess,'embeedings')

print index
embed = tf.nn.embedding_lookup(embeedings,index)

embed = sess.run(embed)

print sess.run(a)
print embed
'''