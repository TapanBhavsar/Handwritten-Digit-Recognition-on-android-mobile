from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import numpy as np
import cv2
import tensorflow as tf
import os
import time

from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

MODEL_NAME = 'model'

def load_dataset(validation_percentage = 0.2, num_classes = 10):
  lines = []
  with open('train-labels.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)

  images = []
  actions = []

  for line in lines:
    images.append(cv2.imread(str(line[0])))
    actions.append(line[1])

  train_images = np.asarray(images)
  action = np.asarray(actions, dtype = np.int32)

  train_labels = np.zeros((len(actions),num_classes),dtype = np.int32)
  for i in range(len(train_labels)):
      a = action[i]
      train_labels[i,a] = 1
  
  validation_images = train_images[-int(validation_percentage * len(train_images)):]
  validation_labels = train_labels[-int(validation_percentage * len(train_labels)):]
  train_images = train_images[:-int(validation_percentage * len(train_images))]
  train_labels = train_labels[:-int(validation_percentage * len(train_labels))]

  lines = []
  with open('test-labels.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)

  images = []
  actions = []

  for line in lines:
    images.append(cv2.imread(str(line[0])))
    actions.append(line[1])

  test_images = np.asarray(images)
  action = np.asarray(actions, dtype = np.int32)

  test_labels = np.zeros((len(actions),num_classes),dtype = np.int32)
  for i in range(len(test_labels)):
        a = action[i]
        test_labels[i,a] = 1

  offset = np.mean(train_images, 0)
  scale = np.std(train_images, 0).clip(min=1)
  train_images = (train_images - offset) / scale

  offset = np.mean(validation_images, 0)
  scale = np.std(validation_images, 0).clip(min=1)
  validation_images = (validation_images - offset) / scale

  offset = np.mean(test_images, 0)
  scale = np.std(test_images, 0).clip(min=1)
  test_images = (test_images - offset) / scale

  return train_images, train_labels, validation_images, validation_labels, test_images, test_labels, num_classes

def build_model(input_val,w,b):

  conv1 = tf.nn.conv2d(input_val,w['w1'],strides = [1,1,1,1], padding = 'SAME')
  conv1 = tf.nn.bias_add(conv1,b['b1'])
  conv1 = tf.nn.relu(conv1)
  pool1 = tf.nn.max_pool(conv1, ksize = (1,2,2,1), strides = [1,2,2,1], padding ='SAME')

  conv2 = tf.nn.conv2d(pool1, w['w2'], strides = [1,1,1,1], padding = 'SAME')
  conv2 = tf.nn.bias_add(conv2,b['b2'])
  conv2 = tf.nn.relu(conv2)
  pool2 = tf.nn.max_pool(conv2, ksize = (1,2,2,1), strides = [1,2,2,1], padding = 'SAME')

  shape = pool2.get_shape().as_list()
  dense = tf.reshape(pool2,[-1,shape[1]*shape[2]*shape[3]])
  dense = tf.nn.dropout(dense, 0.5)
  dense = tf.nn.bias_add(tf.matmul(dense,w['w3']),b['b3'])
  
  return dense

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

def main_function(input_node_name,num_epochs = 100):
  
  print("loading dataset...")
  X_train,y_train,X_valid, y_valid, X_test,Y_test, num_classes = load_dataset()

  x = tf.placeholder(tf.float32,[None,28,28,3], name = input_node_name)
  y = tf.placeholder(tf.int32,[None,num_classes])

  weights = {

    'w1': tf.Variable(tf.random_normal([5,5,3,6],stddev = 0.1)),
    'w2': tf.Variable(tf.random_normal([5,5,6,12],stddev = 0.1)),
    'w3': tf.Variable(tf.random_normal([7*7*12,num_classes],stddev = 0.1)),
  }

  biases = {

    'b1': tf.Variable(tf.random_normal([6],stddev = 0.1)),
    'b2': tf.Variable(tf.random_normal([12],stddev = 0.1)),
    'b3': tf.Variable(tf.random_normal([num_classes],stddev = 0.1)),
  }
  predict = build_model(x,weights,biases)
  prediction = tf.nn.softmax(predict, name = 'output')
  error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predict,labels = y))
  optm = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(error)
  corr = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
  accuracy = tf.reduce_mean(tf.cast(corr,tf.float32))
  saver = tf.train.Saver()
  init = tf.global_variables_initializer()

  sess = tf.Session()
  sess.run(init)
  tf.train.write_graph(sess.graph_def, 'out/', MODEL_NAME + '.pbtxt', True)
  print("Starting training...")
  for epoch in range(num_epochs):
    train_err = 0
    train_acc = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 1024, shuffle=True):
        inputs, targets = batch
        _,err,acc= sess.run([optm,error,accuracy],feed_dict = {x: inputs,y: targets})# apply tensor function
        train_err += err
        train_acc += acc
        train_batches += 1

    val_err = 0
    val_acc = 0
    val_batches = 0
    for batch in iterate_minibatches(X_valid, y_valid, 1024, shuffle=False):
        inputs, targets = batch
        err, acc = sess.run([error,accuracy],feed_dict = {x: inputs,y: targets})
        val_err += err
        val_acc += acc
        val_batches += 1
    print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
    print("  validation accuracy:\t\t{:.2f} %".format(
                    val_acc / val_batches * 100))
  
  save_path = saver.save(sess,'out/' + MODEL_NAME + '.chkp')
  
  test_accuracy = sess.run([accuracy], feed_dict = {x: X_test, y: Y_test})
  print("Test accuracy:\t" + str(test_accuracy[0] * 100))
  sess.close()

def export_model(input_node_names, output_node_name):
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
        'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
        "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_names], [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

def main():
  if not os.path.exists('out'):
    os.mkdir('out')
  input_node_name = 'input'
  output_node_name = 'output'
  main_function(input_node_name = input_node_name)
  export_model(input_node_name, output_node_name)

if __name__ == '__main__':
    main()
