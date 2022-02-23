from locale import MON_2
import os
import shutil
import numpy as np
import funcy as fy
import tensorflow as tf
import mlflow
import matplotlib.pyplot as plt
from pathlib import Path

import usgoc.utils as utils
import usgoc.evaluation.utils as eu
import usgoc.evaluation.models as em
import usgoc.evaluation.datasets as ed
import usgoc.metrics.multi as mm

class DryRunException(Exception):
  pass

def evaluate_single(
  get_model_ctr, get_ds, model_name,
  repeat=0, fold=0, epochs=1000, patience=100,
  convert_mode=None, limit_id=None,
  ds_id="", override=False, dry=False, tensorboard_embeddings=False,
  return_models=False, return_model_paths=False, return_metrics=False,
  return_dims=False, return_ds=False, lazy_return=False):
    if repeat < 0 or fold < 0:
      return None

    try:
      log_dir_base = f"{utils.PROJECT_ROOT}/logs/{ds_id}/{model_name}"
      run = eu.find_inner_run(fold, repeat)
      if run is not None:
        run_id = run.info.run_id
        run_status = run.info.status
        if run_status == "FINISHED" and not override:
          print(
            f"Skipping {ds_id}_repeat{repeat}, {model_name}.",
            f"Existing run: {run_id}.")
          res = ()
          if return_models:
            model_load_fn = lambda: mlflow.keras.load_model(
              f"runs:/{run_id}/models", custom_objects=dict(
                SparseMultiAccuracy=mm.SparseMultiAccuracy))
            if lazy_return:
              res += (model_load_fn,)
            else:
              res += (model_load_fn(),)
          if return_model_paths:
            res += (
              os.path.join(run.info.artifact_uri, "models", "data", "model"),)
          if return_metrics:
            res += (run.data.metrics,)
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
        else:
          raise DryRunException(
            f"Run {run_id} would be overidden. Doing nothing due to dry.")

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
        mlflow.log_params(hp_dict["values"])

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

        test_res = model.evaluate(test_ds, return_dict=True)
        for k, v in test_res.items():
          mlflow.log_metric(f"test_{k}", v, -1)

        print(f"Finished {ds_id}_repeat{repeat}, {model_name} ({run_id}).")

        res = ()
        if return_models:
          if lazy_return:
            res += (lambda: model,)
          else:
            res += (model,)
        if return_model_paths:
          res += (
            os.path.join(run.info.artifact_uri, "models", "data", "model"),)
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
      best_fold_dims = None
      for fold in folds:
        fold_crits = []
        best_repeat_crit = None
        best_repeat_path = None
        best_repeat_dims = None
        for model_path, metrics, dims in fold:
          if isinstance(criterion, str):
            crit = metrics[criterion]
          else:
            crit = np.mean([metrics[c] for c in criterion])
          fold_crits.append(crit)
          if best_repeat_crit is None or crit > best_repeat_crit:
            best_repeat_crit = crit
            best_repeat_path = model_path
            best_repeat_dims = dims
        # Use mean - 1std as fold peformance criterion:
        fold_crit = np.mean(fold_crits) - np.std(fold_crits)
        if best_fold_crit is None or fold_crit > best_fold_crit:
          best_fold_crit = fold_crit
          best_fold_path = best_repeat_path
          best_fold_dims = best_repeat_dims

      model_path = best_fold_path[len("file://"):]
      print(f"{convert_mode} {limit_id}: Selected model {model_path}.")
      shutil.rmtree(target_dir, ignore_errors=True)
      shutil.copytree(model_path, f"{target_dir}/model")
      best_fold_dims["in_enc"] = hypermodel_builder.in_enc
      utils.cache_write(
        f"{target_dir}/dims.json", best_fold_dims, "json")

def aggregate_confusion_matrices(cms, normalize=True):
  cms = fy.lcat(cms)
  m = np.sum(cms, axis=0)
  if normalize:
    m = np.around(m / np.sum(m, axis=1, keepdims=True), 2) * 100
  return m

def get_preds_and_targets(model, ds):
  preds1 = []
  preds2 = []
  targets1 = []
  targets2 = []
  for batch_in, batch_out in ds:
    pred = model(batch_in)
    preds1.append(pred[0])
    preds2.append(pred[1])
    targets1.append(batch_out[0])
    targets2.append(batch_out[1])

  preds1 = tf.concat(preds1, axis=0)
  preds2 = tf.concat(preds2, axis=0)
  targets1 = tf.concat(targets1, axis=0)
  targets2 = tf.concat(targets2, axis=0)
  return (preds1, preds2), (targets1, targets2)

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

  print("loaded", len(cms))

  labels1, labels2 = ed.get_target_label_dims()
  labels1_keys = labels1.keys()
  labels2_keys = labels2.keys()
  print("keys", labels1_keys, labels2_keys)

  for convert_mode, lids in cms.items():
    print(convert_mode, len(lids))
    for limit_id, get_folds in lids.items():
      print(limit_id)
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
        for fold in folds:
          cm1_train, cm2_train = [], []
          cm1_val, cm2_val = [], []
          cm1_test, cm2_test = [], []
          for get_model, get_ds in fold: # getter fns due to "lazy_return"
            model = get_model()
            _, train_ds, val_ds, test_ds = get_ds()
            (preds1, preds2), (tgts1, tgts2) = get_preds_and_targets(model, train_ds)
            m1_train = mm.sparse_multi_confusion_matrix(tgts1, preds1).numpy()
            m2_train = mm.sparse_multi_confusion_matrix(tgts2, preds2).numpy()
            (preds1, preds2), (tgts1, tgts2) = get_preds_and_targets(model, val_ds)
            m1_val = mm.sparse_multi_confusion_matrix(tgts1, preds1).numpy()
            m2_val = mm.sparse_multi_confusion_matrix(tgts2, preds2).numpy()
            (preds1, preds2), (tgts1, tgts2) = get_preds_and_targets(model, test_ds)
            m1_test = mm.sparse_multi_confusion_matrix(tgts1, preds1).numpy()
            m2_test = mm.sparse_multi_confusion_matrix(tgts2, preds2).numpy()

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

      utils.cache(
        lambda: create_plot(cms1["train"], labels1_keys),
        plot_dir / f"{prefix}_train_label1.pdf", format="plot")
      utils.cache(
        lambda: create_plot(cms1["val"], labels1_keys),
        plot_dir / f"{prefix}_val_label1.pdf", format="plot")
      utils.cache(
        lambda: create_plot(cms1["test"], labels1_keys),
        plot_dir / f"{prefix}_test_label1.pdf", format="plot")
      utils.cache(
        lambda: create_plot(cms2["train"], labels2_keys),
        plot_dir / f"{prefix}_train_label2.pdf", format="plot")
      utils.cache(
        lambda: create_plot(cms2["val"], labels2_keys),
        plot_dir / f"{prefix}_val_label2.pdf", format="plot")
      utils.cache(
        lambda: create_plot(cms2["test"], labels2_keys),
        plot_dir / f"{prefix}_test_label2.pdf", format="plot")
