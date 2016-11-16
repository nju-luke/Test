# -*- coding: utf-8 -*-
# @Time    : 16-9-13 上午10:41
# @Author  : Luke
# @Software: PyCharm

import tensorflow as tf
from tensorflow.models.embedding import gen_word2vec as word2vec


class SquareTest(tf.test.TestCase):
    def testSquare(self):
        with self.test_session():
            x = tf.square([2, 3])
            self.assertAllEqual(x.eval(), [4, 9])

    def testSqrt(self):
        with self.test_session():
            x = tf.sqrt(3.)
            print x.eval()


if __name__ == '__main__':
    # tf.test.main()

    (words, counts, words_per_epoch, _epoch, _words, examples,
     labels) = word2vec.skipgram(filename="/media/luke/工作/Wiki/wiki00_chs_cut",#"data/text8",
                                 batch_size=32,
                                 window_size=5,
                                 min_count=5,
                                 subsample=0.001)
    session = tf.Session()
    (vocab_words, vocab_counts,
     words_per_epoch) = session.run([words, counts, words_per_epoch])
    print "Down"