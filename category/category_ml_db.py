from __future__ import print_function

import tensorflow as tf
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import MySQLdb

# Open database connection
db = MySQLdb.connect("localhost","bboroccu","1234","demotest" )

# prepare a cursor object using cursor() method
cursor = db.cursor()

# execute SQL query using execute() method.
cursor.execute("SELECT item, enr_cate_id from itemcsv limit 100")

# Fetch a single row using fetchone() method.
rows = cursor.fetchall()
vectorizer = CountVectorizer(min_df=1, tokenizer=lambda x: list(x), ngram_range=(2, 4))
datas = list(rows)
categorynames = []
categoryids = []
for row in datas:
    categorynames.append(row[0])
    categoryids.append(row[1])
# disconnect from server
db.close()
analyze = vectorizer.build_analyzer()

ve = vectorizer.fit_transform(categorynames)
x_data = ve.toarray()
y_data = categoryids

X = tf.placeholder(tf.float32, [None, 1])
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

hypothesis = tf.matmul(W, x_data) + b

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for step in range(2001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))

print(sess.run(hypothesis, feed_dict={X:5}))
print(sess.run(hypothesis, feed_dict={X:2.5}))



