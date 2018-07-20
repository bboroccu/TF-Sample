#!/usr/bin/python
from __future__ import print_function

import tensorflow as tf
import numpy as np

csv_file = tf.train.string_input_producer(['20160630.txt'], name='filename_queue')
textReader = tf.TextLineReader()
_,line = textReader.read(csv_file)
item, caid, midid = tf.decode_csv(line, record_defaults=[[""], [], []], field_delim = '\t')

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.string)
Y = tf.placeholder(tf.float32)

hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a= tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(2001):
        sess.run(train, feed_dict={X: item, Y: caid})
        if step % 20 == 0:
            print(step, sess.run(cost, feed_dict={X: item, Y: item}), sess.run(W), sess.run(b))
    #for i in range(100):
    #    item_value, caid_value, midid_value = sess.run([item, caid, midid])
    #    print(item_value, caid_value)
    coord.request_stop()
    coord.join(threads)