import os
from pickle import HIGHEST_PROTOCOL
import shutil
import numpy as np
import funcy as fy
import tensorflow as tf
import keras_tuner as kt
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path
import yaml

import usgoc.utils as utils
import usgoc.evaluation.utils as eu
import usgoc.evaluation.models as em
import usgoc.evaluation.datasets as ed
import usgoc.models.utils as mu
import usgoc.metrics.multi as mm
import usgoc.datasets.unsafe_go as dataset
import usgoc.postprocessing.explain as explain

class DryRunException(Exception):
  pass

calib_config_name = "conformal_calibration_configs.yml"
size_hist_name_fn = lambda prefix: f"{prefix}conformal_set_histograms.yml"

def evaluate_single_conformal(
  model, train_ds, val_ds, test_ds,
  alphas=[0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]):
  calibration_configs = mu.multi_calibrate_conformal(model, val_ds, alphas)
  print("Computed conformal calibration configs:", calibration_configs)

  mlflow.log_dict(calibration_configs, calib_config_name)

  def log_conf(ds, prefix=""):
    preds, targets = mu.multi_predict_conformal_with_targets(
      model, ds, calibration_configs)
    histograms = mu.multi_conformal_histograms(preds)
    mlflow.log_dict(histograms, f"{prefix}conformal_set_histograms.yml")
    metrics = mu.multi_conformal_metrics(preds, targets)
    print(f"Computed conformal {prefix}metrics:", metrics)
    for k, v in metrics.items():
      mlflow.log_metric(f"{prefix}{k}", v, -1)
    return histograms

  train_hist = log_conf(train_ds)
  val_hist = log_conf(val_ds, "val_")
  test_hist = log_conf(test_ds, "test_")
  return calibration_configs, (train_hist, val_hist, test_hist)

def evaluate_single(
  get_model_ctr, get_ds, model_name,
  repeat=0, fold=0, epochs=1000, patience=100,
  convert_mode=None, limit_id=None,
  ds_id="", override=False, override_conformal=False,
  dry=False, tensorboard_embeddings=False,
  return_models=False, return_model_paths=False,
  return_calibration_configs=False, return_metrics=False,
  return_size_histograms=False,
  return_dims=False, return_ds=False, lazy_return=False, ignore_status=False,
  delete_runs=False):
    if return_model_paths:
      assert get_model_ctr.is_keras, "Cannot return model paths for non-keras models."

    if repeat < 0 or fold < 0:
      return None

    try:
      log_dir_base = f"{utils.PROJECT_ROOT}/logs/{ds_id}/{model_name}"
      run = eu.find_inner_run(fold, repeat)
      if run is not None:
        run_id = run.info.run_id
        run_status = run.info.status
        artifact_uri = run.info.artifact_uri[len("file://"):]
        if (run_status == "FINISHED" or ignore_status) and not override and not delete_runs:
          def model_load_fn():
            if not get_model_ctr.is_keras:
              model, _ = get_model_ctr()()
              dims, train_ds, val_ds, test_ds = get_ds()
              model.fit(train_ds)
              return model
            return mlflow.keras.load_model(
              f"runs:/{run_id}/models", custom_objects=dict(
                  SparseMultiAccuracy=mm.SparseMultiAccuracy))
          model = None
          calib_configs = None
          hists = None
          calib_config_path = os.path.join(artifact_uri, calib_config_name)
          hist_paths = fy.lmap(
            lambda pre: os.path.join(artifact_uri, size_hist_name_fn(pre)),
            ["", "val_", "test_"])
          if override_conformal or not os.path.exists(calib_config_path):
            print(f"Overriding conformal metrics {ds_id}_repeat{repeat}, {model_name}.",
                  f"Calibration config path: {calib_config_path}.",
                  f"Existing run: {run_id}")
            with mlflow.start_run(run_id, nested=True) as r:
              _, train_ds, val_ds, test_ds = get_ds()
              model = model_load_fn()
              calib_configs, hists = evaluate_single_conformal(model, train_ds, val_ds, test_ds)
              run = r
          else:
            print(
              f"Skipping {ds_id}_repeat{repeat}, {model_name}.",
              f"Existing run: {run_id}.")
          res = ()
          if return_models:
            if lazy_return:
              if model is not None:
                res += (lambda: model,)
              else:
                res += (model_load_fn,)
            else:
              res += (model_load_fn(),)
          if return_model_paths:
            res += (
              os.path.join(run.info.artifact_uri, "models", "data", "model"),)
          if return_calibration_configs:
            if calib_configs is None:
              with open(calib_config_path, "r") as f:
                res += (yaml.unsafe_load(f),)
            else:
              res += (calib_configs,)
          if return_metrics:
            res += (run.data.metrics,)
          if return_size_histograms:
            if hists is None:
              for pre in ["", "val_", "test_"]:
                with open(os.path.join(
                  artifact_uri, size_hist_name_fn(pre)), "r") as f:
                  res += (yaml.unsafe_load(f),)
            else:
              res += hists
          if return_ds:
            if lazy_return:
              res += (get_ds,)
            else:
              res += get_ds()
          elif return_dims:
            dims_load_fn = lambda: ed.get_dims(fold,
              convert_mode=convert_mode, limit_id=limit_id)
            if lazy_return:
              res += (dims_load_fn,)
            else:
              res += (dims_load_fn(),)
          if len(res) == 1:
            res = res[0]
          elif len(res) == 0:
            res = None
          return res
        elif not dry:
          mlflow.delete_run(run_id)
          shutil.rmtree(f"{log_dir_base}/{run_id}", ignore_errors=True)
          print(
            f"Deleting {run_status} {ds_id}_repeat{repeat}, {model_name}.",
            f"Existing run: {run_id}.")
          if delete_runs:
            return
          else:
            eu.invalidate_inner_run_cache()
        else:
          raise DryRunException(
            f"Run {run_id} ({run_status}) would be overidden. Doing nothing due to dry.")

      if delete_runs:
        raise DryRunException(
          f"Nonexistent {ds_id}_repeat{repeat}, {model_name} cannot be deleted.")

      if lazy_return:
        print(f"WARNING: True lazy returns impossible due to cache miss.")

      parent_tags = mlflow.active_run().data.tags
      model_ctr = get_model_ctr()
      mlflow.tensorflow.autolog(log_models=False)

      with mlflow.start_run(
        run_name=f"fold{fold}_repeat{repeat}",
        nested=True) as run:
        for k, v in parent_tags.items():
          if not k.startswith("mlflow") and k != "type":
            mlflow.set_tag(k, v)
        mlflow.set_tag("fold", fold)
        mlflow.set_tag("repeat", repeat)
        mlflow.set_tag("type", "inner")
        run_id = run.info.run_id
        print(f"Starting {ds_id}_repeat{repeat}, {model_name} ({run_id})...")

        if dry:
          raise DryRunException(f"Dry run {run_id}. Doing nothing.")

        dims, train_ds, val_ds, test_ds = get_ds()
        model, hp_dict = model_ctr()
        if hp_dict is not None:
          mlflow.log_params(hp_dict["values"])

        if isinstance(model, tf.keras.Model):
          log_dir = f"{log_dir_base}/{run_id}"
          stop_early = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=patience,
            restore_best_weights=True)
          callbacks = [stop_early]

          if tensorboard_embeddings:
            tb = tf.keras.callbacks.TensorBoard(
              log_dir=log_dir,
              histogram_freq=10,
              embeddings_freq=10,
              write_graph=True,
              update_freq="batch")
            callbacks.append(tb)

          model.fit(
            train_ds,
            validation_data=val_ds,
            callbacks=callbacks,
            verbose=2, epochs=epochs)
          mlflow.keras.log_model(model, "models")
        else:
          model.fit(train_ds)
          train_res = model.evaluate(train_ds, return_dict=True)
          for k, v in train_res.items():
            mlflow.log_metric(k, v, -1)
          val_res = model.evaluate(val_ds, return_dict=True)
          for k, v in val_res.items():
            mlflow.log_metric(f"val_{k}", v, -1)

        test_res = model.evaluate(test_ds, return_dict=True)
        for k, v in test_res.items():
          mlflow.log_metric(f"test_{k}", v, -1)

        calib_configs = evaluate_single_conformal(model, train_ds, val_ds, test_ds)

        print(f"Finished {ds_id}_repeat{repeat}, {model_name} ({run_id}).")

        res = ()
        if return_models:
          if lazy_return:
            res += (lambda: model,)
          else:
            res += (model,)
        else:
          tf.keras.backend.clear_session()
        if return_model_paths:
          res += (
            os.path.join(run.info.artifact_uri, "models", "data", "model"),)
        if return_calibration_configs:
          res += (calib_configs,)
        if return_metrics:
          res += (run.data.metrics,)
        if return_ds:
          if lazy_return:
            res += (lambda: (dims, train_ds, val_ds, test_ds),)
          else:
            res += (dims, train_ds, val_ds, test_ds)
        elif return_dims:
          if lazy_return:
            res += (lambda: dims,)
          else:
            res += (dims,)
        if len(res) == 1:
          res = res[0]
        elif len(res) == 0:
          res = None
        return res
    except DryRunException as e:
      print(str(e))
      return None

def evaluate_fold(
  hypermodel_builder, fold=0, repeats=eu.REPEATS_MAX,
  start_repeat=0, repeat=None, convert_mode=None, limit_id=None,
  ds_name="", tuner_convert_mode=None, tuner_limit_id=None,
  keep_nesting=False, **kwargs):

  ds_id = f"{ds_name}/{convert_mode}_{limit_id}_fold{fold}"
  get_ds = utils.memoize(lambda: ed.get_encoded(
    hypermodel_builder.in_enc, fold=fold,
    convert_mode=convert_mode, limit_id=limit_id))

  if tuner_convert_mode == convert_mode and tuner_limit_id == limit_id:
    tune_ds_id = ds_id
    get_tune_ds = get_ds
  else:
    tune_ds_id = f"{ds_name}/{tuner_convert_mode}_{tuner_limit_id}_fold{fold}"
    get_tune_ds = lambda: ed.get_encoded(
      hypermodel_builder.in_enc, fold=fold,
      convert_mode=tuner_convert_mode, limit_id=tuner_limit_id)

  if not issubclass(hypermodel_builder, kt.HyperModel):
    def get_model_ctr():
      return lambda: (hypermodel_builder(), None)
    get_model_ctr.is_keras = False
  else:
    @utils.memoize
    def get_model_ctr():
      tune_dims, train_ds, val_ds, test_ds = get_tune_ds()
      tuner_hypermodel = hypermodel_builder(**tune_dims)

      mlflow.tensorflow.autolog(disable=True)
      tuner = em.tune_hyperparams(
        tuner_hypermodel, train_ds, val_ds, ds_id=tune_ds_id)
      tuner.search_space_summary()
      if tuner_limit_id is None:
        return lambda: em.get_best_model(tuner)
      else:
        dims = get_ds()[0]
        hypermodel = hypermodel_builder(**dims)

        def model_ctr():
          best_hps = em.get_best_hps(tuner)
          return hypermodel.build(best_hps), best_hps.get_config()

        return model_ctr
    get_model_ctr.is_keras = True

  if getattr(hypermodel_builder, "is_deterministic", False):
    repeat = 0

  if repeat is not None:
    res = evaluate_single(
      get_model_ctr, get_ds, hypermodel_builder.name, repeat,
      fold=fold, convert_mode=convert_mode, limit_id=limit_id, ds_id=ds_id,
      **kwargs)
    if keep_nesting:
      res = [res]
    return res

  return [
    evaluate_single(
      get_model_ctr, get_ds, hypermodel_builder.name, i,
      fold=fold, convert_mode=convert_mode, limit_id=limit_id, ds_id=ds_id,
      **kwargs)
    for i in range(start_repeat, repeats)]

def summarize_inner_runs():
  pass

def evaluate_limit_id(
  hypermodel_builder, convert_mode=None, limit_id=None,
  folds=eu.FOLDS_MAX,
  tuner_convert_mode=None, tuner_limit_id=None,
  start_fold=0, fold=None, ds_name=None, keep_nesting=False,
  lazy_folds=False, **kwargs):
  p_tuner_convert_mode = tuner_convert_mode
  p_tuner_limit_id = tuner_limit_id

  def iterate_folds():
    tuner_convert_mode = p_tuner_convert_mode
    tuner_limit_id = p_tuner_limit_id
    mname = hypermodel_builder.name
    run = eu.find_outer_run(convert_mode, limit_id, mname)
    if run is not None:
      run_id = run.info.run_id
    else:
      run_id = None

    with mlflow.start_run(
      run_id=run_id,
      run_name=f"{convert_mode}_{limit_id}_{mname}"):
      mlflow.set_tag("model", mname)
      mlflow.set_tag("dataset", ds_name)
      mlflow.set_tag("convert_mode", convert_mode)
      mlflow.set_tag("limit_id", limit_id)
      mlflow.set_tag("type", "outer")
      if getattr(hypermodel_builder, "is_deterministic", False):
        mlflow.set_tag("deterministic", True)

      print(f"Starting outer {ds_name}/{convert_mode}/{limit_id} for {mname}...")

      if tuner_convert_mode is not None:
        print(f"(Tuned with HPs for convert mode {tuner_convert_mode})")
      else:
        tuner_convert_mode = convert_mode

      if tuner_limit_id is not None:
        print(f"(Tuned with HPs for limit {tuner_limit_id})")
      else:
        tuner_limit_id = limit_id

      mlflow.set_tag("tuner_convert_mode", tuner_convert_mode)
      mlflow.set_tag("tuner_limit_id", tuner_limit_id)

      if fold is not None:
        res = evaluate_fold(
          hypermodel_builder, fold=fold,
          convert_mode=convert_mode, limit_id=limit_id, ds_name=ds_name,
          tuner_convert_mode=tuner_convert_mode, tuner_limit_id=tuner_limit_id,
          keep_nesting=keep_nesting, **kwargs)
        if keep_nesting:
          res = [res]
      else:
        res = [
          evaluate_fold(
            hypermodel_builder, fold=i, convert_mode=convert_mode,
            limit_id=limit_id, ds_name=ds_name,
            tuner_convert_mode=tuner_convert_mode, tuner_limit_id=tuner_limit_id,
            keep_nesting=keep_nesting, **kwargs)
          for i in range(start_fold, folds)]

      summarize_inner_runs()
      print(f"Completed outer {ds_name}/{limit_id} for {mname}.")

      return res

  if lazy_folds:
    return iterate_folds
  else:
    return iterate_folds()

def evaluate_convert_mode(
  hypermodel_builder, convert_mode=None,
  limit_ids=ed.evaluate_limit_ids,
  limit_id=None, keep_nesting=False, **kwargs):

  if limit_id is not None:
    res = evaluate_limit_id(
      hypermodel_builder, convert_mode, limit_id,
      keep_nesting=keep_nesting, **kwargs)
    return {limit_id: res} if keep_nesting else res

  return {
    limit_id: evaluate_limit_id(
      hypermodel_builder, convert_mode, limit_id,
      keep_nesting=keep_nesting, **kwargs)
    for limit_id in limit_ids}

def cast_hypermodel_buider(hypermodel_builder):
  if isinstance(hypermodel_builder, str):
    assert hypermodel_builder in em.models,\
        f"Unknown model {hypermodel_builder}."
    return em.models[hypermodel_builder]

  return hypermodel_builder

def evaluate(
  hypermodel_builder,
  ds_name=ed.dataset_names[0],
  convert_modes=ed.convert_modes,
  convert_mode=None,
  keep_nesting=False,
  experiment_suffix="", **kwargs):
  hypermodel_builder = cast_hypermodel_buider(hypermodel_builder)
  mlflow.set_experiment(ds_name + experiment_suffix)
  print(f"Opening experiment {hypermodel_builder.name}{experiment_suffix}...")

  try:
    if convert_mode is not None:
      res = evaluate_convert_mode(
        hypermodel_builder, convert_mode, ds_name=ds_name,
        keep_nesting=keep_nesting, **kwargs)
      return {convert_mode: res} if keep_nesting else res

    return {
      convert_mode: evaluate_convert_mode(
        hypermodel_builder, convert_mode, ds_name=ds_name,
        keep_nesting=keep_nesting, **kwargs)
      for convert_mode in convert_modes}
  finally:
    print(f"Closing experiment {hypermodel_builder.name}{experiment_suffix}.")

def export_best(
  hypermodel_builder,
  criterion=["val_accuracy", "test_accuracy"],
  **kwargs):
  models_dir = f"{utils.PROJECT_ROOT}/exported_models"
  hypermodel_builder = cast_hypermodel_buider(hypermodel_builder)

  utils.cache_write(
    f"{models_dir}/target_label_dims.json",
    ed.get_target_label_dims(),
    "json")

  model_name = hypermodel_builder.name
  kwargs["return_model_paths"] = True
  kwargs["return_calibration_configs"] = True
  kwargs["return_metrics"] = True
  kwargs["return_dims"] = True
  kwargs["dry"] = True
  kwargs["keep_nesting"] = True
  cms = evaluate(hypermodel_builder, **kwargs)
  for convert_mode, lids in cms.items():
    for limit_id, folds in lids.items():
      target_dir = f"{models_dir}/{convert_mode}_{limit_id}/{model_name}"
      best_fold_crit = None
      best_fold_path = None
      best_fold_calib = None
      best_fold_dims = None
      for fold in folds:
        fold_crits = []
        best_repeat_crit = None
        best_repeat_path = None
        best_repeat_calib = None
        best_repeat_dims = None
        for model_path, calib_configs, metrics, dims in fold:
          if isinstance(criterion, str):
            crit = metrics[criterion]
          else:
            crit = np.mean([metrics[c] for c in criterion])
          fold_crits.append(crit)
          if best_repeat_crit is None or crit > best_repeat_crit:
            best_repeat_crit = crit
            best_repeat_path = model_path
            best_repeat_calib = calib_configs
            best_repeat_dims = dims
        # Use mean - 1std as fold peformance criterion:
        fold_crit = np.mean(fold_crits) - np.std(fold_crits)
        if best_fold_crit is None or fold_crit > best_fold_crit:
          best_fold_crit = fold_crit
          best_fold_path = best_repeat_path
          best_fold_calib = best_repeat_calib
          best_fold_dims = best_repeat_dims

      model_path = best_fold_path[len("file://"):]
      print(f"{convert_mode} {limit_id}: Selected model {model_path}.")
      shutil.rmtree(target_dir, ignore_errors=True)
      shutil.copytree(model_path, f"{target_dir}/model")
      best_fold_dims["in_enc"] = hypermodel_builder.in_enc
      utils.cache_write(
        f"{target_dir}/dims.json", best_fold_dims, "json")
      utils.cache_write(
        f"{target_dir}/conformal_calibration_configs.yml", best_fold_calib, "yaml")

def aggregate_confusion_matrices(cms, normalize=True):
  cms = fy.lcat(cms)
  m = np.sum(cms, axis=0)
  if normalize:
    m = np.around(m / np.sum(m, axis=1, keepdims=True), 2) * 100
  return m

def export_confusion_matrices(
  hypermodel_builder, normalize=True, **kwargs):
  matrix_dir = Path(f"{utils.PROJECT_ROOT}/results/confusion_matrices_raw")
  plot_dir = Path(f"{utils.PROJECT_ROOT}/results/confusion_matrices_plot")
  utils.make_dir(matrix_dir)
  utils.make_dir(plot_dir)
  hypermodel_builder = cast_hypermodel_buider(hypermodel_builder)
  model_name = hypermodel_builder.name

  kwargs["return_models"] = True
  kwargs["return_ds"] = True
  kwargs["dry"] = True
  kwargs["keep_nesting"] = True
  kwargs["lazy_return"] = True
  kwargs["lazy_folds"] = True
  cms = evaluate(hypermodel_builder, **kwargs)

  labels1, labels2 = ed.get_target_label_dims()
  labels1_keys = labels1.keys()
  labels2_keys = labels2.keys()

  for convert_mode, lids in cms.items():
    for limit_id, get_folds in lids.items():
      prefix = f"{convert_mode}_{limit_id}_{model_name}"

      @utils.memoize
      def compute_cms():
        folds = get_folds()
        cms1, cms2 = {
          "train": [],
          "val": [],
          "test": [],
        }, {
          "train": [],
          "val": [],
          "test": [],
        }
        i = 0
        n = len(folds)
        for fold in folds:
          cm1_train, cm2_train = [], []
          cm1_val, cm2_val = [], []
          cm1_test, cm2_test = [], []
          m = len(fold)
          nm = n * m
          for get_model, get_ds in fold: # getter fns due to "lazy_return"
            i += 1
            print(f"Computing predictions {i}/{nm}...")
            model = get_model()
            _, train_ds, val_ds, test_ds = get_ds()
            (preds1, preds2), (tgts1, tgts2) = mu.predict_with_targets(model, train_ds)
            m1_train = mm.sparse_multi_confusion_matrix(tgts1, preds1).numpy()
            m2_train = mm.sparse_multi_confusion_matrix(tgts2, preds2).numpy()
            (preds1, preds2), (tgts1, tgts2) = mu.predict_with_targets(model, val_ds)
            m1_val = mm.sparse_multi_confusion_matrix(tgts1, preds1).numpy()
            m2_val = mm.sparse_multi_confusion_matrix(tgts2, preds2).numpy()
            (preds1, preds2), (tgts1, tgts2) = mu.predict_with_targets(model, test_ds)
            m1_test = mm.sparse_multi_confusion_matrix(tgts1, preds1).numpy()
            m2_test = mm.sparse_multi_confusion_matrix(tgts2, preds2).numpy()
            tf.keras.backend.clear_session()

            cm1_train.append(m1_train)
            cm1_val.append(m1_val)
            cm1_test.append(m1_test)
            cm2_train.append(m2_train)
            cm2_val.append(m2_val)
            cm2_test.append(m2_test)
          cms1["train"].append(cm1_train)
          cms1["val"].append(cm1_val)
          cms1["test"].append(cm1_test)
          cms2["train"].append(cm2_train)
          cms2["val"].append(cm2_val)
          cms2["test"].append(cm2_test)
        return cms1, cms2

      print(f"Computing confusion matrices for {prefix}...")
      cms1, cms2 = utils.cache(
        lambda: compute_cms(), matrix_dir / f"{prefix}.pickle", format="pickle")

      print(f"Creating aggregated confusion matrix plots...")
      def create_plot(cms, labels):
        m = aggregate_confusion_matrices(cms, normalize)
        return utils.draw_confusion_matrix(m.astype(int), labels, False)

      for s in ["train", "val", "test"]:
        utils.cache(
          lambda: create_plot(cms1[s], labels1_keys),
          plot_dir / f"{prefix}_{s}_label1.pdf", format="plot")
        utils.cache(
          lambda: create_plot(cms2[s], labels2_keys),
          plot_dir / f"{prefix}_{s}_label2.pdf", format="plot")

def export_feature_importances(
  hypermodel_builder, **kwargs):
  fimps_dir = Path(f"{utils.PROJECT_ROOT}/results/feature_importance_raw")
  plot_dir = Path(f"{utils.PROJECT_ROOT}/results/feature_importance_plot")
  utils.make_dir(fimps_dir)
  utils.make_dir(plot_dir)
  hypermodel_builder = cast_hypermodel_buider(hypermodel_builder)
  model_name = hypermodel_builder.name
  in_enc = hypermodel_builder.in_enc

  kwargs["return_models"] = True
  kwargs["return_ds"] = True
  kwargs["dry"] = True
  kwargs["keep_nesting"] = True
  kwargs["lazy_return"] = True
  kwargs["lazy_folds"] = True
  cms = evaluate(hypermodel_builder, **kwargs)

  labels1, labels2 = ed.get_target_label_dims()
  labels1_keys = list(labels1.keys())
  labels2_keys = list(labels2.keys())

  for convert_mode, lids in cms.items():
    for limit_id, get_folds in lids.items():
      prefix = f"{convert_mode}_{limit_id}_{model_name}"

      @utils.memoize
      def compute_fimps():
        folds = get_folds()
        dims, fimps1, fimps2 = [], [], []
        i = 0
        n = len(folds)
        for fold in folds:
          dim, fimp1, fimp2 = [], [], []
          m = len(fold)
          nm = n * m
          for get_model, get_ds in fold:  # getter fns due to "lazy_return"
            i += 1
            print(f"Computing importances {i}/{nm}...")
            model = get_model()
            d, _, _, test_ds = get_ds()
            f1, f2 = explain.compute_importances(model, test_ds)
            dim.append(d)
            fimp1.append(f1)
            fimp2.append(f2)
            tf.keras.backend.clear_session()

          dims.append(dim)
          fimps1.append(fimp1)
          fimps2.append(fimp2)
        return dims, fimps1, fimps2

      print(f"Computing feature importances for {prefix}...")
      dims, fimps1, fimps2 = utils.cache(
        lambda: compute_fimps(), fimps_dir / f"{prefix}.pickle", format="pickle")

      print(f"Creating aggregated feature importance plots...")

      def create_plot(dims, fimps1, fimps2):
        fimps1 = fy.lmap(explain.group_feature_importance, fy.cat(fimps1))
        fimps2 = fy.lmap(explain.group_feature_importance, fy.cat(fimps2))
        merged_dims, remap_idxs = dataset.merge_dims(fy.lcat(dims))
        fimps1 = dataset.apply_remap_idxs(
          fimps1, remap_idxs,
          merged_dims["node_label_count"], in_enc, with_marked_idx=True)
        fimps2 = dataset.apply_remap_idxs(
          fimps2, remap_idxs,
          merged_dims["node_label_count"], in_enc, with_marked_idx=True)
        fimps1 = np.sum(fimps1, 0)
        fimps2 = np.sum(fimps2, 0)
        return utils.draw_feature_importance_chart([
          ("Label 1", labels1_keys + ["Combined"], fimps1),
          ("Label 2", labels2_keys + ["Combined"], fimps2)
        ], dataset.dims_to_labels(merged_dims, in_enc), show=False)

      utils.cache(
        lambda: create_plot(dims, fimps1, fimps2),
        plot_dir / f"{prefix}_test.pdf", format="plot")

def delete_orphaned_runs(
  model, ds_name=ed.dataset_names[0],
  experiment_suffix="", dry=False, **kwargs):
  mlflow.set_experiment(ds_name + experiment_suffix)
  print("Finding orphans...")
  orphan_runs = eu.find_orphaned_runs()
  for run in orphan_runs:
    run_id = run.info.run_id
    run_status = run.info.status
    tags = run.data.tags
    run_name = tags["mlflow.runName"]
    parent = tags["mlflow.parentRunId"]
    lid = tags.get("limit_id", "-")
    m = tags.get("model", "-")
    print(
      f"Deleting {run_status} orphan {run_name} ({run_id} child of {parent}). "
      f"Model: {m}, Limit ID: {lid}.")
    if dry:
      print("- Doing nothing due to dry.")
    else:
      mlflow.delete_run(run_id)
