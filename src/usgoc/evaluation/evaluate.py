from typing import List
import shutil
import tensorflow as tf
import mlflow

import usgoc.utils as utils
import usgoc.evaluation.models as em
import usgoc.evaluation.datasets as ed
import usgoc.metrics.multi as mm

mlflow.set_tracking_uri(f"file:{utils.PROJECT_ROOT}/mlruns")

FOLDS_MAX = 10
REPEATS_MAX = 3

class DryRunException(Exception):
  pass

def find_outer_run(convert_mode, limit_id, model_name):
  eid = mlflow.tracking.fluent._get_experiment_id()
  runs = mlflow.search_runs(
    [eid],
    " and ".join([
      "tags.type = 'outer'",
      f"tags.convert_mode = '{convert_mode}'",
      f"tags.limit_id = '{limit_id}'",
      f"tags.model = '{model_name}'"]),
    max_results=1, output_format="list")
  if len(runs) == 0:
    return None

  return runs[0]

def find_inner_runs(fold=None, repeat=None) -> List[mlflow.entities.Run]:
  eid = mlflow.tracking.fluent._get_experiment_id()
  parent_id = mlflow.active_run().info.run_id
  max_results = 1
  conditions = [
    "tags.type = 'inner'",
    f"tags.mlflow.parentRunId = '{parent_id}'"]
  if fold is not None:
    conditions.append(f"tags.fold = '{fold}'")
  else:
    max_results = FOLDS_MAX
  if repeat is not None:
    conditions.append(f"tags.repeat = '{repeat}'")
  else:
    max_results *= REPEATS_MAX
  query = " and ".join(conditions)
  runs = mlflow.search_runs(
    [eid], query, max_results=max_results, output_format="list")

  return runs

def find_inner_run(fold, repeat):
  runs = find_inner_runs(fold, repeat)
  if len(runs) == 0:
    return None
  return runs[0]

def evaluate_single(
  get_model_ctr, get_ds, model_name,
  repeat=0, fold=0, epochs=1000, patience=100,
  convert_mode=None, limit_id=None,
  ds_id="", override=False, dry=False, tensorboard_embeddings=False,
  return_models=True):
    if repeat < 0 or fold < 0:
      return None

    try:
      log_dir_base = f"{utils.PROJECT_ROOT}/logs/{ds_id}/{model_name}"
      run = find_inner_run(fold, repeat)
      if run is not None:
        run_id = run.info.run_id
        run_status = run.info.status
        if run_status == "FINISHED" and not override:
          print(
            f"Skipping {ds_id}_repeat{repeat}, {model_name}.",
            f"Existing run: {run_id}.")
          if dry or not return_models:
            return None
          return mlflow.keras.load_model(
            f"runs:/{run_id}/models",
            custom_objects=dict(SparseMultiAccuracy=mm.SparseMultiAccuracy))
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

        if not return_models:
          return None
        return model
    except DryRunException as e:
      print(str(e))
      return None

def evaluate_fold(
  hypermodel_builder, fold=0, repeats=REPEATS_MAX,
  start_repeat=0, repeat=None, convert_mode=None, limit_id=None,
  ds_name="", tuner_convert_mode=None, tuner_limit_id=None,
  **kwargs):

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
    return evaluate_single(
      get_model_ctr, get_ds, hypermodel_builder.name, repeat,
      fold=fold, convert_mode=convert_mode, limit_id=limit_id, ds_id=ds_id,
      **kwargs)

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
  folds=FOLDS_MAX,
  tuner_convert_mode=None, tuner_limit_id=None,
  start_fold=0, fold=None, ds_name=None, **kwargs):
  mname = hypermodel_builder.name
  run = find_outer_run(convert_mode, limit_id, mname)
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
        **kwargs)
    else:
      res = [
        evaluate_fold(
          hypermodel_builder, fold=i, convert_mode=convert_mode,
          limit_id=limit_id, ds_name=ds_name,
          tuner_convert_mode=tuner_convert_mode, tuner_limit_id=tuner_limit_id,
          **kwargs)
        for i in range(start_fold, folds)]

    summarize_inner_runs()
    print(f"Completed outer {ds_name}/{limit_id} for {mname}.")

    return res

def evaluate_convert_mode(
  hypermodel_builder, convert_mode=None,
  limit_ids=ed.evaluate_limit_ids,
  limit_id=None, **kwargs):

  if limit_id is not None:
    return evaluate_limit_id(
      hypermodel_builder, convert_mode, limit_id, **kwargs)

  return {
    limit_id: evaluate_limit_id(
      hypermodel_builder, convert_mode, limit_id, **kwargs)
    for limit_id in limit_ids}

def evaluate(
  hypermodel_builder,
  ds_name=ed.dataset_names[0],
  convert_modes=ed.convert_modes,
  convert_mode=None,
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
      return evaluate_convert_mode(
        hypermodel_builder, convert_mode, ds_name=ds_name, **kwargs)

    return {
      convert_mode: evaluate_convert_mode(
        hypermodel_builder, convert_mode, ds_name=ds_name, **kwargs)
      for convert_mode in convert_modes}
  finally:
    print(f"Closing experiment {hypermodel_builder.name}{experiment_suffix}.")
