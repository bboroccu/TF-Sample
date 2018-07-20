import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from matplotlib import pyplot as plt
import random

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)
cost = tf.reduce_mean(tf.reduce_sum(-Y*tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(cost)
init =tf.initialize_all_variables()

training_epoch = 25
display_step = 1
batch_size = 100
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epoch):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        avg_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})/total_batch
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
            print(sess.run(b))

    print ("Optimization Finished!")

    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

    print ("TEST!")
    r = random.randint(0, mnist.test.num_examples - 1)
    print "Label : ", sess.run(tf.argmax(mnist.test.labels[r:r+1], 1))
    print "Prediction : ", sess.run(tf.argmax(hypothesis, 1), {X: mnist.test.images[r:r+1]})

    plt.imshow(mnist.test.images[r:r+1].reshape(28, 28), cmap='Greys', interpolation='nearest')
    plt.show()