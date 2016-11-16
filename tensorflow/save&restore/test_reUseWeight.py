# -*- coding: utf-8 -*-
# @Time    : 16-8-30 上午10:58
# @Author  : Luke
# @Software: PyCharm
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

'''
v1 = tf.Variable(..., name='v1')
v2 = tf.Variable(..., name='v2')

# Pass the variables as a dict:
saver = tf.train.Saver({'v1': v1, 'v2': v2})

# Or pass them as a list.
saver = tf.train.Saver([v1, v2])
# Passing a list is equivalent to passing a dict with the variable op names
# as keys:
saver = tf.train.Saver({v.op.name: v for v in [v1, v2]})

'''


class test_reuse:
    def __init__(self):
        self.epochs = 100
        n_observations = 100
        xs = np.linspace(-3, 3, n_observations)
        self.xs = np.linspace(-3, 3, n_observations)
        self.ys = 3 * xs + 2 + np.random.uniform(-0.5, 0.5, n_observations)
        # plt.plot(xs, self.ys)
        # plt.show()
        # with tf.variable_scope("foo"):
        #     self.W = tf.get_variable("W", [1])
        #     self.b = tf.get_variable("b", [1])
        self.W = tf.Variable(np.random.rand())
        self.b = tf.Variable(np.random.rand())
        self.xs_placeholder = tf.placeholder(tf.float32)
        self.ys_placeholder = tf.placeholder(tf.float32)
        self.y_pred = self.xs_placeholder * self.W + self.b
        self.loss = tf.reduce_sum(tf.pow(self.y_pred - self.ys_placeholder, 2)) / (n_observations - 1)
        self.lr = 0.005
        self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)

    def train(self):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            prev_train_cost = np.inf
            flag = 10
            for epoch in xrange(self.epochs):
                for (x, y) in zip(self.xs, self.ys):
                    sess.run(self.optimizer, feed_dict={self.xs_placeholder: x, self.ys_placeholder: y})

                training_cost = sess.run(
                    self.loss, feed_dict={self.xs_placeholder: self.xs, self.ys_placeholder: self.ys})

                prev_train_cost = training_cost
                if epoch % 10 == 0:
                    print(training_cost, sess.run(self.W), sess.run(self.b))

            # with tf.variable_scope("foo", reuse=True):
            #     W = tf.get_variable("W")

            # tf.train.Saver([self.W,self.b]).save(sess, 'myModel')
            tf.train.Saver({"W": self.W, "b": self.b}).save(sess, 'myModel')

    def pred(self, x):
        # with tf.Session() as sess:
        #     # y = sess.run(self.y_pred,feed_dict={self.xs_placeholder:x})
        #     y = sess.run(self.W * x + self.b)        self.lr = 0.001


        sess = tf.Session()
        saver = tf.train.Saver({"W": self.W, "b": self.b})
        saver.restore(sess, 'myModel')
        # with tf.variable_scope("foo",reuse=True):
        # W = tf.get_variable("W")
        y = sess.run(self.W * x + self.b)
        # print sess.run(W)
        print y


class test1:
    def train(self):
        sess = tf.Session()
        # with tf.variable_scope("set1"):
        W = tf.get_variable("W", [1])
        v = tf.get_variable("v", [1])
        # u = tf.get_variable("u", [1])

        sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver([W, v])
        saver.save(sess, 'myModel')
        print sess.run(W)

    def get(self):
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, 'myModel')
        vars = tf.trainable_variables()
        varsList = [v.name for v in vars if "W" in v.name]
        print varsList
        print sess.run(varsList[0])


class test2:
    def save(self):
        with tf.variable_scope("set1"):
            v1 = tf.get_variable("v1", [1])
            v2 = tf.get_variable("v2", [1])
            # cons = tf.constant(range(5),name="cons")

        init = tf.initialize_all_variables()

        saver = tf.train.Saver()

        '''
        指定存储v2后则只存储了v2，再读取时将没有除v2以外的信息
        需要保证加载的变量全部都被存储，否则将会报错

        saver 可以保存的只有variable，constant是不可以保存的

        另外查字典是应该将字符转化为数字，这样就不需要对应的字典转化了
        '''

        with tf.Session() as sess:
            sess.run(init)
            print v2.eval(sess)
            save_path = "model.ckpt"
            saver.save(sess, save_path)
            # saver.restore(sess, save_path)
            print("Model saved.")

    @classmethod
    def restore(self):
        with tf.variable_scope("set1", reuse=True):
            v1 = tf.get_variable("v1", [1])
            v2 = tf.get_variable("v2", [1])
            cons = tf.constant(range(5), name="cons")

        saver = tf.train.Saver([v2])
        # saver = tf.train.Saver([v2,cons])
        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            save_path = "model.ckpt"
            saver.restore(sess, save_path)
            print("Model restored.")
            print sess.run(v2), "\n"
            # print sess.run(v2),"\n",cons


if __name__ == '__main__':
    # test_class = test_reuse()
    # test_class.train()
    # test_class.pred(3)
    # test_c = test1()
    # test_c.train()
    # test_c.get()

    test2 = test2()
    test2.save()
    test2.restore()

'''
save保存变量是可以选择
restore以后
(1)用trainable_variable(),v.name中会包含“：0”这样的字段，可以利用规则提取对应的变量
(2)用with tf.variable_scope() 和 tf.get_variable() 设置reuse=true
    此时restore可以取代initialization来使用

tf.get_variable() 用于获取已经建立的变量
tf.Variable()     用于建立新的变量              所以，在同一个scope中，get的变量与先前建立的一样，而Variable()的变量中scope将被重命名为foo_v1
'''
