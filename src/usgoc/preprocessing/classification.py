import numpy as np

def one_hot(classes, class_count=2):
  multilabel = isinstance(class_count, tuple)
  assert not multilabel, "Multilabel classification not yet supported."
  res = np.zeros((len(classes), class_count))
  res[np.arange(len(classes)), classes] = 1

  return res
