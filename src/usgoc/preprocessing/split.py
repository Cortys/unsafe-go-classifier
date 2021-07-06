import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

def make_holdout_split(holdout, objects, strat_labels=None):
  if holdout == 0:
    return objects, [], strat_labels
  else:
    train_split, test_split = train_test_split(
      np.arange(objects.size),
      test_size=holdout,
      stratify=strat_labels)

    if strat_labels is not None:
      strat_labels = strat_labels[train_split]

    return objects[train_split], objects[test_split], strat_labels

def make_kfold_splits(n, objects, strat_labels=None):
  if strat_labels is None:
    kfold = KFold(n_splits=n, shuffle=True)
  else:
    kfold = StratifiedKFold(n_splits=n, shuffle=True)

  for train_split, test_split in kfold.split(objects, strat_labels):
    yield (
      objects[train_split],
      objects[test_split],
      strat_labels[train_split] if strat_labels is not None else None
    )

def make_model_selection_splits(
  train_o, strat_o=None,
  inner_k=None, inner_holdout=0.1):
  if inner_k is None:
    train_i, val_i, _ = make_holdout_split(
      inner_holdout, train_o, strat_o)
    return [dict(
      train=train_i,
      validation=val_i)]
  else:
    return [
      dict(train=train_i, validation=val_i)
      for train_i, val_i, _ in make_kfold_splits(
        inner_k, train_o, strat_o)]

def make_splits(
  size,
  outer_k=10, inner_k=None,
  outer_holdout=None, inner_holdout=0.1,
  strat_labels=None):
  all_idxs = np.arange(size)

  if outer_k is None:
    train_o, test_o, strat_o = make_holdout_split(
      outer_holdout, all_idxs, strat_labels)
    return [dict(
      test=test_o,
      model_selection=make_model_selection_splits(train_o, strat_o))]
  else:
    return [
      dict(
        test=test_o,
        model_selection=make_model_selection_splits(train_o, strat_o))
      for train_o, test_o, strat_o in make_kfold_splits(
        outer_k, all_idxs, strat_labels)]
