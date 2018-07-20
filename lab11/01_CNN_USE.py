import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.examples.tutorials.mnist import input_data

batch_size = 128
test_size = 256

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w1, w2, w3, w4, w_out, p_keep_conv, p_keep_hidden):
    l1a = tf.nn.relu(tf.nn.conv2d(X, w1, strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, w2, strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, w3, strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    l3 = tf.reshape(l3, [-1, w4.get_shape().as_list()[0]])
    l3 = tf.nn.dropout(l3, p_keep_conv)

    l4 = tf.nn.relu(tf.matmul(l3, w4))
    l4 = tf.nn.dropout(l4, p_keep_hidden)
    pyx = tf.matmul(l4, w_out)
    return pyx

training_epoch = 15
display_step = 1

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)
teX = teX.reshape(-1, 28, 28, 1)

X = tf.placeholder("float", [None, 28, 28, 1])
Y = tf.placeholder("float", [None, 10])

w1 = init_weights([3, 3, 1, 32])
w2 = init_weights([3, 3, 32, 64])
w3 = init_weights([3, 3, 64, 128])
w4 = init_weights([128*4*4, 625])
w_out = init_weights([625, 10])

p_keep_conv = tf.placeholder("float")
p_keep_hidden = tf.placeholder("float")
py_x = model(X, w1, w2, w3, w4, w_out, p_keep_conv, p_keep_hidden)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
init = tf.initialize_all_variables()
predict_op = tf.argmax(py_x, 1)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, '/Users/bboroccu/Documents/workspace/PycharmProjects/TensorFlowExam/lab11/cnn_session')
    tf.initialize_all_variables().run()
    training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size))
    test_indices = np.arange(len(teX))
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:test_size]
    print(np.argmax(teY[test_indices], axis=1))
    print sess.run(predict_op, feed_dict={X: teX[test_indices], Y: teY[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})
    print(np.mean(np.argmax(teY[test_indices], axis=1) == sess.run(predict_op, feed_dict={X: teX[test_indices], Y: teY[test_indices], p_keep_conv: 1.0, p_keep_hidden: 1.0})))
    """
    im = mpimg.imread("7.jpg")
    gray = rgb2gray(im)
    arr_data = []

    for xPos in xrange(28):
        for yPos in xrange(28):
            temp = gray.__getitem__(xPos)
            value = temp[yPos]
            arr_data.append(value / 255.0)
    tempX = np.array([arr_data], dtype='float32')
    #p_val = sess.run(p, feed_dict={X: gray, 1.0})
    p_val = sess.run(predict_op, feed_dict={X: teX[test_indices], p_keep_hidden: 1.0})
    print p_val
    fig = plt.figure(figsize=(4, 2))
    pred = p_val[0]
    subplot = fig.add_subplot(1, 1, 1)
    subplot.set_xticks(range(10))
    subplot.set_xlim(-0.5, 9.5)
    subplot.set_ylim(0, 1)
    subplot.bar(range(10), pred, align='center')
    plt.show()
    """