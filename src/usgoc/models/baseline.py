import numpy as np

class MajorityClassifier:
  name = "Majority"
  in_enc = "raw"
  is_deterministic = True

  def __init__(self, **kwargs):
    self.trained = False
    self.output = None

  def fit(self, ds, **kwargs):
    _, (labels1, labels2) = ds
    n = len(labels1)
    self.output = (
      np.log(np.bincount(labels1, minlength=11) / n, dtype=np.float32),
      np.log(np.bincount(labels2, minlength=11) / n, dtype=np.float32))
    self.trained = True

  def predict(self, ds, **kwargs):
    if isinstance(ds, tuple) and len(ds) == 2:
      ds = ds[0]
    n = len(ds)
    o1, o2 = self.output
    return np.broadcast_to(o1, (n, len(o1))), np.broadcast_to(o2, (n, len(o2)))

  def __call__(self, ds, **kwargs):
    return self.predict(ds)

  def evaluate(self, ds, **kwargs):
    assert self.trained, "Model has to be trained before predicting."
    _, (targets1, targets2) = ds
    n = len (targets1)
    o1, o2 = self.output
    p1, p2 = np.argmax(o1), np.argmax(o2)
    l1_res = targets1 == p1
    l2_res = targets2 == p2
    l1_acc = np.sum(l1_res) / n
    l2_acc = np.sum(l2_res) / n
    acc = np.sum(l1_res & l2_res) / n
    return dict(
      label1_accuracy=l1_acc,
      label2_accuracy=l2_acc,
      accuracy=acc
    )
