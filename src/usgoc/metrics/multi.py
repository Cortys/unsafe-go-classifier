import tensorflow as tf
from tensorflow import keras

class SparseMultiAccuracy(keras.metrics.Metric):
  def __init__(self, name="label", dtype=None, group=None):
    print("init", name, dtype, group)
    assert group is not None, "Missing accuracy group."
    super().__init__(name, dtype)
    assert name not in group["labels"]
    group["labels"].add(name)
    group["head"] = name
    self._group = group
    self._label = name
    self._metric = keras.metrics.Mean(name=f"{name}_accuracy")

  @staticmethod
  def create_group(name="accuracy"):
    group = dict(
      head=None,
      labels=set(),
      marked=set(),
      reset=set(),
      comb_acc=None,
      metric=keras.metrics.Mean(name=name)
    )
    return group

  def reset_state(self):
    group = self._group
    group["reset"].add(self.name)
    self._metric.reset_state()

    if len(group["reset"]) == len(group["labels"]):
      group["marked"] = set()
      group["comb_acc"] = None
      group["reset"] = set()
      group["metric"].reset_state()

  def update_state(self, y_true, y_pred, sample_weight=None):
    group = self._group
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
    group = self._group
    assert len(group["marked"]) == 0, "Forbidden partial mark."
    assert len(group["reset"]) == 0, "Forbidden partial reset."

    res = dict()
    res[self._metric.name] = self._metric.result()

    if group["head"] == self._label:
      res[group["metric"].name] = group["metric"].result()

    return res
