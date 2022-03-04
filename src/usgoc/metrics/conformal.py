import numpy as np

def set_accuracy(y_true, y_pred):
  return np.mean([yt in yp for yt, yp in zip(y_true, y_pred)])

def multi_set_accuracy(y_trues, y_preds):
  comb_accs = np.ones_like(y_trues[0], dtype=bool)
  for yts, yps in zip(y_trues, y_preds):
    comb_accs &= [yt in yp for yt, yp in zip(yts, yps)]

  return np.mean(comb_accs)

def hist_to_dict(hist):
  sizes, freqs = hist
  return dict(zip(sizes, freqs))

def set_size_histogram(sets, as_dict=True):
  set_sizes = [len(s) for s in sets]
  sizes, freqs = np.unique(set_sizes, return_counts=True)
  hist = sizes, freqs
  return hist_to_dict(hist) if as_dict else hist

def set_size_mean(sets):
  return np.mean([len(s) for s in sets])

def set_size_median(sets):
  return np.median([len(s) for s in sets])
