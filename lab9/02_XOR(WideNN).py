import tensorflow as tf
import numpy as np

xy = np.loadtxt('train.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_uniform([2, 10], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([10, 1], -1.0, 1.0))

b1 = tf.Variable(tf.zeros([10]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

cost = tf.reduce_mean(-Y * tf.log(hypothesis) - (1-Y) * tf.log(1-hypothesis))
optimizer = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X:x_data, Y:y_data})
        if step % 100 == 0:
            print sess.run(cost, feed_dict={X:x_data, Y:y_data})

    answer = tf.equal(tf.floor(hypothesis + 0.5), Y)
    accuracy = tf.reduce_mean(tf.cast(answer, "float"))
    print sess.run([hypothesis], feed_dict={X:x_data, Y:y_data})
    print "Accuracy : ", accuracy.eval({X:x_data, Y:y_data}) * 100, "%"