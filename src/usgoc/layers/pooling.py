import tensorflow as tf
import tensorflow.keras as keras
from abc import ABCMeta, abstractmethod

class SegmentPooling(keras.layers.Layer, metaclass=ABCMeta):
  def __init__(self):
    super().__init__()

  def call(self, input):
    X = input["X"]
    graph_idx = input["graph_idx"]
    N = tf.shape(input["n"])[0]
    y = self.pool(X, graph_idx, num_segments=N)
    return y

  @staticmethod
  @abstractmethod
  def pool(X, graph_idx, num_segments): pass

class MeanPooling(SegmentPooling):
  pool = staticmethod(tf.math.unsorted_segment_mean)

class SumPooling(SegmentPooling):
  pool = staticmethod(tf.math.unsorted_segment_sum)

class MaxPooling(SegmentPooling):
  pool = staticmethod(tf.math.unsorted_segment_max)

class MinPooling(SegmentPooling):
  pool = staticmethod(tf.math.unsorted_segment_min)

class SoftmaxPooling(keras.layers.Layer):
  def __init__(self, uniform_attention=True):
    super().__init__()
    self.uniform_attention = True

  def get_config(self):
    return dict(
      **super().get_config(),
      uniform_attention=self.uniform_attention)

  def call(self, input):
    if "X_att" in input:
      X = input["X"]
      X_att = input["X_att"]
    elif self.uniform_attention:
      X = input["X"]
      X_att = tf.reshape(X[:, 0], [-1, 1])
      X = X[:, 1:]
    else:
      X, X_att = tf.split(input["X"], 2, axis=1)
    graph_idx = input["graph_idx"]
    N = tf.shape(input["n"])[0]

    X_att = tf.nn.softmax(X_att, axis=0)
    X = X_att * X
    y = tf.math.unsorted_segment_sum(
      X, graph_idx, num_segments=N)
    return y

def merge_attention(inputs):
  input, input_att = inputs

  return {**input, "X_att": input_att["X"]}
