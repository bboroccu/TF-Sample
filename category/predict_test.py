import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import requests
from flask import Flask, request
from flask import jsonify
app = Flask(__name__)
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "/Users/bboroccu/Documents/workspace/PycharmProjects/TensorFlowExam/category/runs", "Checkpoint directory from training run")
tf.flags.DEFINE_string("run_dir", "1499759371", "train predictions")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, FLAGS.run_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
# Evaluation
  #  ==================================================
checkpoint_file = tf.train.latest_checkpoint(os.path.join(FLAGS.checkpoint_dir, FLAGS.run_dir, "checkpoints"))
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]
@app.route('/predict', methods=['GET', 'POST'])
def getPredict():
  x_raw = []
  y_test = []
  post_id = request.args.get('data')
  post_id = post_id.decode('utf-8')
  print post_id
  x_raw.append(post_id)

  x_test = np.array(list(vocab_processor.transform(x_raw)))



  # Collect the predictions here
  all_predictions = []
  # Generate batches for one epoch
  batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
  for x_test_batch in batches:
    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
    all_predictions = np.concatenate([all_predictions, batch_predictions])
  result = all_predictions[0]
  myDict = {'code': result, 'name': result}
  return jsonify(myDict)
if __name__ == "__main__":
  app.run(host='0.0.0.0', debug=True)