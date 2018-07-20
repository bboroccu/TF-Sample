import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

batch_size = 128
test_size = 784

tf.reset_default_graph()

num_filters1 = 32

x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, num_filters1], stddev = 0.1))
h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
b_conv1 = tf.Variable(tf.constant(0.1, shape=[num_filters1]))
h_conv1_cutoff = tf.nn.relu(h_conv1 + b_conv1)
h_pool1 = tf.nn.max_pool(h_conv1_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

num_filters2 = 64

W_conv2 = tf.Variable(tf.truncated_normal([5, 5, num_filters1, num_filters2], stddev = 0.1))
h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
b_conv2 = tf.Variable(tf.constant(0.1, shape=[num_filters2]))
h_conv2_cutoff = tf.nn.relu(h_conv2 + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2_cutoff, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*num_filters2])
num_units1 = 7*7*num_filters2
num_units2 = 1024

w2 = tf.Variable(tf.truncated_normal([num_units1, num_units2]))
b2 = tf.Variable(tf.constant(0.1, shape=[num_units2]))
hidden2 = tf.nn.relu(tf.matmul(h_pool2_flat, w2) + b2)

keep_prob = tf.placeholder(tf.float32)
hidden2_drop = tf.nn.dropout(hidden2, keep_prob)

w0 = tf.Variable(tf.zeros([num_units2, 10]))
b0 = tf.Variable(tf.zeros([10]))
k = tf.matmul(hidden2_drop, w0) + b0
p = tf.nn.softmax(k)

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
teX = teX.reshape(-1, 784)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
saver = tf.train.Saver()
saver.restore(sess, '/Users/bboroccu/Documents/workspace/PycharmProjects/TensorFlowExam/lab11/cnn_session')

print 'reload has been done'

for i in xrange(1):
    training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size))
    test_indices = np.arange(len(teX))
    np.random.shuffle(test_indices)
    test_indices = test_indices[0:test_size]

    im = mpimg.imread("5.jpg")
    gray = rgb2gray(im)
    arr_data = []

    for xPos in xrange(28):
        for yPos in xrange(28):
            temp = gray.__getitem__(xPos)
            value = temp[yPos]
            arr_data.append((255.0 - value) / 255.0)
    tempX = np.array([arr_data], dtype='float32')
    p_val = sess.run(p, feed_dict={x: tempX, keep_prob: 1.0})
    #p_val = sess.run(p, feed_dict={x: teX[test_indices], keep_prob: 1.0})
    print p_val
    print np.argmax(p_val, axis=1)
    fig = plt.figure(figsize=(4, 2))
    pred = p_val[0]
    subplot = fig.add_subplot(1, 1, 1)
    subplot.set_xticks(range(10))
    subplot.set_xlim(-0.5, 9.5)
    subplot.set_ylim(0, 1)
    subplot.bar(range(10), pred, align='center')

    conv1_vals, cutoff1_vals = sess.run(
        [h_conv1, h_conv1_cutoff], feed_dict={x: tempX, keep_prob: 1.0})

    fig = plt.figure(figsize=(16,4))

    for f in range(num_filters1):
        subplot = fig.add_subplot(4, 16, f+1)
        subplot.set_xticks([])
        subplot.set_yticks([])
        subplot.imshow(conv1_vals[0,:,:,f], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()
