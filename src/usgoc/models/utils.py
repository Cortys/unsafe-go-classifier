import numpy as np
from collections.abc import Iterable

import usgoc.postprocessing.calibration as cal
import usgoc.postprocessing.conformal as conf
import usgoc.metrics.conformal as conf_met

def predict_with_targets(model, ds):
  preds1 = []
  preds2 = []
  targets1 = []
  targets2 = []

  if isinstance(ds, tuple) and len(ds) == 2:
    ds = [ds]

  for input, targets in ds:
    pred1, pred2 = model(input)
    target1, target2 = targets
    preds1.append(pred1)
    preds2.append(pred2)
    targets1.append(target1)
    targets2.append(target2)

  preds1 = np.concatenate(preds1)
  preds2 = np.concatenate(preds2)
  targets1 = np.concatenate(targets1)
  targets2 = np.concatenate(targets2)

  return (preds1, preds2), (targets1, targets2)

def multi_calibrate_conformal(model, cal_ds, alphas=[0.1]):
  (preds1, preds2), (targets1, targets2) = predict_with_targets(model, cal_ds)
  t1 = cal.platt_scaling(preds1, targets1)
  t2 = cal.platt_scaling(preds2, targets2)
  preds1 /= t1
  preds2 /= t2
  scores1 = conf.adaptive_calibration_scores(preds1, targets1)
  scores2 = conf.adaptive_calibration_scores(preds2, targets2)
  return {alpha: dict(
    t1=t1, t2=t2,
    qhat1=conf.qhat(scores1, alpha),
    qhat2=conf.qhat(scores2, alpha)
  ) for alpha in alphas}

def calibrate_conformal(model, cal_ds, alpha=0.1):
  return multi_calibrate_conformal(model, cal_ds, [alpha])[alpha]

def to_set_prediction(
  preds1, preds2, *, t1, t2, qhat1, qhat2, with_preds=False):
  preds1 = preds1 / t1
  preds2 = preds2 / t2
  sets1 = conf.adaptive_sets(preds1, qhat1)
  sets2 = conf.adaptive_sets(preds2, qhat2)
  if with_preds:
    return sets1, sets2, preds1, preds2
  return sets1, sets2

def predict_conformal(model, ds, **calibration_config):
  preds = model.predict(ds)
  return to_set_prediction(*preds, **calibration_config)

def conformal_metrics(sets, targets):
  sets1, sets2 = sets
  targets1, targets2 = targets

  return dict(
    label1_accuracy_conf=conf_met.set_accuracy(targets1, sets1),
    label2_accuracy_conf=conf_met.set_accuracy(targets2, sets2),
    accuracy_conf=conf_met.multi_set_accuracy(targets, sets),
    label1_mean_size_conf=conf_met.set_size_mean(sets1),
    label2_mean_size_conf=conf_met.set_size_mean(sets2),
    label1_median_size_conf=conf_met.set_size_median(sets1),
    label2_median_size_conf=conf_met.set_size_median(sets2)
  )

def conformal_histograms(sets):
  sets1, sets2 = sets
  return conf_met.set_size_histogram(sets1), conf_met.set_size_histogram(sets2)

# Batch functions for multiple calibrations (different alphas) at once:

def multi_predict_conformal(model, ds, calibration_configs):
  preds = model.predict(ds)
  return {
    k: to_set_prediction(*preds, **v)
    for k, v in calibration_configs.items()
  }

def multi_predict_conformal_with_targets(model, ds, calibration_configs):
  preds, targets = predict_with_targets(model, ds)
  return {
    k: to_set_prediction(*preds, **v)
    for k, v in calibration_configs.items()
  }, targets

def multi_conformal_metrics(multi_preds, targets):
  return {
    f"{met}_{alpha}": val
    for alpha, sets in multi_preds.items()
    for met, val in conformal_metrics(sets, targets).items()
  }

def multi_conformal_histograms(multi_preds):
  return {
    alpha: conformal_histograms(sets)
    for alpha, sets in multi_preds.items()
  }
