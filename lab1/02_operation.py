from __future__ import print_function
import tensorflow as tf

sess = tf.Session()

a = tf.constant(2)
b = tf.constant(3)

c = a+b
print(a)
print(b)
print(c)

print(a+b)

print(sess.run(a))
print(sess.run(b))
print(sess.run(c))
print(sess.run(a+b))