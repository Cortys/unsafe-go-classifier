from typing import List
import shutil
import tensorflow as tf
import mlflow

import usgoc.utils as utils
import usgoc.evaluation.models as em
import usgoc.evaluation.datasets as ed
import usgoc.metrics.multi as mm

# mlflow.set_tracking_uri("http://127.0.0.1:1234")
mlflow.set_tracking_uri(f"file:{utils.PROJECT_ROOT}/mlruns")

FOLDS_MAX = 10
REPEATS_MAX = 3

def find_outer_run(limit_id, model_name):
  eid = mlflow.tracking.fluent._get_experiment_id()
  runs = mlflow.search_runs(
    [eid],
    f"tags.limit_id = '{limit_id}' and tags.model = '{model_name}'",
    max_results=1, output_format="list")
  if len(runs) == 0:
    return None

  return runs[0]

def find_inner_runs(fold=None, repeat=None) -> List[mlflow.entities.Run]:
  eid = mlflow.tracking.fluent._get_experiment_id()
  parent_id = mlflow.active_run().info.run_id
  max_results = 1
  conditions = [
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
  tuner, train_ds, val_ds, test_ds,
  model_name, repeat=0,
  fold=0, epochs=1000, patience=100, limit_id=None,
  ds_id="", override=False, tensorboard_embeddings=False):
    if repeat < 0 or fold < 0:
      return None

    log_dir_base = f"{utils.PROJECT_ROOT}/logs/{ds_id}/{model_name}"
    mlflow.tensorflow.autolog(log_models=False)
    run = find_inner_run(fold, repeat)
    if run is not None:
      run_id = run.info.run_id
      if run.info.status == "FINISHED" and not override:
        print(
          f"Skipping {ds_id}_repeat{repeat}, {model_name}.",
          f"Existing run: {run_id}.")
        return mlflow.keras.load_model(
          f"runs:/{run_id}/models",
          custom_objects=dict(SparseMultiAccuracy=mm.SparseMultiAccuracy))
      else:
        mlflow.delete_run(run_id)
        shutil.rmtree(f"{log_dir_base}/{run_id}", ignore_errors=True)

    with mlflow.start_run(
      run_name=f"fold{fold}_repeat{repeat}",
      nested=True) as run:
      model, hp_dict = em.get_best_model(tuner)
      mlflow.set_tag("fold", fold)
      mlflow.set_tag("repeat", repeat)
      mlflow.log_params(hp_dict["values"])
      run_id = run.info.run_id
      log_dir = f"{log_dir_base}/{run_id}"

      stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience,
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
      return model

def evaluate_fold(
  hypermodel_builder, fold=0, repeats=REPEATS_MAX,
  start_repeat=0, limit_id=None, ds_name="", **kwargs):

  ds_id = f"{ds_name}/{limit_id}_fold{fold}"
  dims, train_ds, val_ds, test_ds = ed.get_encoded(
    hypermodel_builder.in_enc, fold=fold, limit_id=limit_id)
  hypermodel = hypermodel_builder(**dims)

  mlflow.tensorflow.autolog(disable=True)
  tuner = em.tune_hyperparams(
    hypermodel, train_ds, val_ds, ds_id=ds_id)
  tuner.search_space_summary()

  if "repeat" in kwargs:
    return evaluate_single(
      tuner, train_ds, val_ds, test_ds,
      hypermodel.name, fold=fold, limit_id=limit_id, ds_id=ds_id,
      **kwargs)

  return [
    evaluate_single(
      tuner, train_ds, val_ds, test_ds, hypermodel.name, i,
      fold=fold, limit_id=limit_id, ds_id=ds_id, **kwargs)
    for i in range(start_repeat, repeats)]

def summarize_inner_runs():
  inner_runs = find_inner_runs()

  for run in inner_runs:
    print(run)

def evaluate_limit_id(
  hypermodel_builder, limit_id=None, folds=FOLDS_MAX,
  start_fold=0, ds_name=None, **kwargs):
  run = find_outer_run(limit_id, hypermodel_builder.name)
  if run is not None:
    run_id = run.info.run_id
  else:
    run_id = None

  with mlflow.start_run(
    run_id=run_id,
    run_name=f"{limit_id}_{hypermodel_builder.name}"):
    mlflow.set_tag("model", hypermodel_builder.name)
    mlflow.set_tag("dataset", ds_name)
    mlflow.set_tag("limit_id", limit_id)

    if "fold" in kwargs:
      res = evaluate_fold(
        hypermodel_builder, limit_id=limit_id, ds_name=ds_name,
        **kwargs)
    else:
      res = [
        evaluate_fold(
          hypermodel_builder, fold=i, limit_id=limit_id,
          ds_name=ds_name, **kwargs)
        for i in range(start_fold, folds)]

    summarize_inner_runs()

    return res

def evaluate(
  hypermodel_builder,
  ds_name=ed.dataset_names[0],
  limit_ids=ed.evaluate_limit_ids,
  experiment_suffix="", **kwargs):

  mlflow.set_experiment(ds_name + experiment_suffix)

  if "limit_id" in kwargs:
    return evaluate_limit_id(
      hypermodel_builder, ds_name=ds_name, **kwargs)

  return {
    limit_id: evaluate_limit_id(
      hypermodel_builder, limit_id, ds_name=ds_name, **kwargs)
    for limit_id in limit_ids}
