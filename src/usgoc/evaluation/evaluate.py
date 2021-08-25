import os
import shutil
import numpy as np
import tensorflow as tf
import mlflow

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
  return_models=False, return_model_paths=False, return_metrics=False):
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
            res += (mlflow.keras.load_model(
              f"runs:/{run_id}/models", custom_objects=dict(
                SparseMultiAccuracy=mm.SparseMultiAccuracy)),)
          if return_model_paths:
            res += (
              os.path.join(run.info.artifact_uri, "models", "data", "model"),)
          if return_metrics:
            res += (run.data.metrics,)
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
          res += (model,)
        if return_model_paths:
          res += (
            os.path.join(run.info.artifact_uri, "models", "data", "model"),)
        if return_metrics:
          res += (run.data.metrics,)
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
  start_fold=0, fold=None, ds_name=None, keep_nesting=False, **kwargs):
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

def evaluate(
  hypermodel_builder,
  ds_name=ed.dataset_names[0],
  convert_modes=ed.convert_modes,
  convert_mode=None,
  keep_nesting=False,
  experiment_suffix="", **kwargs
):
  if isinstance(hypermodel_builder, str):
    assert hypermodel_builder in em.models,\
        f"Unknown model {hypermodel_builder}."
    hypermodel_builder = em.models[hypermodel_builder]

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
  if isinstance(hypermodel_builder, str):
    model_name = hypermodel_builder
  else:
    model_name = hypermodel_builder.name
  kwargs["return_model_paths"] = True
  kwargs["return_metrics"] = True
  kwargs["dry"] = True
  kwargs["keep_nesting"] = True
  cms = evaluate(hypermodel_builder, **kwargs)
  for convert_mode, lids in cms.items():
    for limit_id, folds in lids.items():
      target_dir = f"{models_dir}/{convert_mode}_{limit_id}/{model_name}"
      best_fold_crit = None
      best_fold_path = None
      for fold in folds:
        fold_crits = []
        best_repeat_crit = None
        best_repeat_path = None
        for model_path, metrics in fold:
          if isinstance(criterion, str):
            crit = metrics[criterion]
          else:
            crit = np.mean([metrics[c] for c in criterion])
          fold_crits.append(crit)
          if best_repeat_crit is None or crit > best_repeat_crit:
            best_repeat_crit = crit
            best_repeat_path = model_path
        # Use mean - 1std as fold peformance criterion:
        fold_crit = np.mean(fold_crits) - np.std(fold_crits)
        if best_fold_crit is None or fold_crit > best_fold_crit:
          best_fold_crit = fold_crit
          best_fold_path = best_repeat_path

      model_path = best_fold_path[len("file://"):]
      print(f"{convert_mode} {limit_id}: Selected model {model_path}.")
      shutil.rmtree(target_dir, ignore_errors=True)
      shutil.copytree(model_path, target_dir)
