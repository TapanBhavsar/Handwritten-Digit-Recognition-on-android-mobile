from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time

from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
from scipy.misc import imsave
import tensorflow as tf
import numpy as np
import csv

IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].
  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    #data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = 255 - data
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, 1)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
  return labels

# Extract it into np arrays.
train_data = extract_data('train-images-idx3-ubyte.gz', 60000)
train_labels = extract_labels('train-labels-idx1-ubyte.gz', 60000)
test_data = extract_data('t10k-images-idx3-ubyte.gz', 10000)
test_labels = extract_labels('t10k-labels-idx1-ubyte.gz', 10000)

if not os.path.isdir("train-images"):
   os.makedirs("train-images")

if not os.path.isdir("test-images"):
   os.makedirs("test-images")

# process train data
with open("train-labels.csv", 'wb') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(train_data)):
    imsave("train-images/" + str(i) + ".jpg", train_data[i][:,:,0])
    writer.writerow(["train-images/" + str(i) + ".jpg", train_labels[i]])

# repeat for test data
with open("test-labels.csv", 'wb') as csvFile:
  writer = csv.writer(csvFile, delimiter=',', quotechar='"')
  for i in range(len(test_data)):
    imsave("test-images/" + str(i) + ".jpg", test_data[i][:,:,0])
    writer.writerow(["test-images/" + str(i) + ".jpg", test_labels[i]])
