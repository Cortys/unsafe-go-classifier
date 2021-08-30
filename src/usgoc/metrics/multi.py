import uuid
import tensorflow as tf
from tensorflow import keras
# import tensorflow_addons as tfa

_groups = dict()

class SparseMultiAccuracy(keras.metrics.Metric):
  def __init__(self, name="label", label=None, dtype=None, group=None):
    assert group is not None, "Missing accuracy group."

    if label is not None:
      name = label

    super().__init__(name, dtype)
    group = SparseMultiAccuracy.create_group(*group)
    self._group = group
    group = self.get_group()
    group["labels"].add(name)
    group["head"] = name
    self._label = name
    self._metric = keras.metrics.Mean(name=f"{name}_accuracy")

  def get_config(self):
    return dict(
      **super().get_config(),
      group=self._group,
      label=self._label)

  @staticmethod
  def create_group(name="accuracy", id=None):
    if id is None:
      id = uuid.uuid4().hex
    group_key = (name, id)

    if group_key in _groups:
      return group_key

    group = dict(
      id=id,
      name=name,
      head=None,
      labels=set(),
      marked=set(),
      reset=set(),
      comb_acc=None,
      metric=keras.metrics.Mean(name=name)
    )
    _groups[group_key] = group
    return group_key

  def get_group(self):
    return _groups[self._group]

  def reset_state(self):
    group = self.get_group()
    group["reset"].add(self.name)
    self._metric.reset_state()

    if len(group["reset"]) == len(group["labels"]):
      group["marked"] = set()
      group["comb_acc"] = None
      group["reset"] = set()
      group["metric"].reset_state()

  def update_state(self, y_true, y_pred, sample_weight=None):
    group = self.get_group()
    assert self.name not in group["marked"], "Illegal double mark."
    assert len(group["reset"]) == 0, "Forbidden partial reset."

    y_true = tf.reshape(y_true, [-1])
    y_pred_max = tf.argmax(y_pred, -1, output_type=y_true.dtype)
    acc = y_pred_max == y_true
    group["marked"].add(self.name)
    if group["comb_acc"] is None:
      group["comb_acc"] = acc
    else:
      group["comb_acc"] &= acc
    self._metric.update_state(acc)

    if len(group["marked"]) == len(group["labels"]):
      group["metric"].update_state(group["comb_acc"])
      group["marked"] = set()
      group["comb_acc"] = None

  def result(self):
    group = self.get_group()
    assert len(group["marked"]) == 0, "Forbidden partial mark."
    assert len(group["reset"]) == 0, "Forbidden partial reset."

    res = dict()
    res[self._metric.name] = self._metric.result()

    if group["head"] == self._label:
      res[group["metric"].name] = group["metric"].result()

    return res

# class SparseMultiBinaryConfusionMatrix(tfa.metrics.MultiLabelConfusionMatrix):
#   def update_state(self, y_true, y_pred, sample_weight=None):
#     y_true = tf.reshape(y_true, [-1])
#     y_true = tf.one_hot(y_true, self.num_classes, dtype=tf.int32)
#     y_pred_max = tf.argmax(y_pred, -1, output_type=tf.int32)
#     y_pred = tf.one_hot(y_pred_max, self.num_classes, dtype=tf.int32)
#     super().update_state(y_true, y_pred, sample_weight)
#
#   def reset_state(self):
#     self.reset_states()

def sparse_multi_confusion_matrix(y_true, y_pred):
  y_true = tf.reshape(y_true, [-1])
  y_pred_max = tf.argmax(y_pred, -1, output_type=tf.int32)
  return tf.math.confusion_matrix(y_true, y_pred_max)
