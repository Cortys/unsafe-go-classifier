import tensorflow as tf
import mlflow

import usgoc.evaluation.models as em
import usgoc.evaluation.datasets as ed

mlflow.set_tracking_uri("file:/app/mlruns")
mlflow_client = None

def get_client():
  global mlflow_client
  if mlflow_client is None:
    mlflow_client = mlflow.tracking.MlflowClient()
  return mlflow_client

def single_evaluate(
  tuner, train_ds, val_ds, test_ds,
  model_name, repeat=0,
  fold=0, epochs=1000, patience=100, limit_id=None,
  experiment_suffix=""):
    ds_name = ed.dataset_names[0]
    ds_id = f"{limit_id}_fold{fold}"
    mlflow.set_experiment(ds_name + experiment_suffix)
    mlflow.tensorflow.autolog(log_models=False)
    with mlflow.start_run():
      model, hp_dict = em.get_best_model(tuner)
      mlflow.set_tag("model", model_name)
      mlflow.set_tag("dataset", ds_name)
      mlflow.set_tag("limit_id", limit_id)
      mlflow.set_tag("fold", fold)
      mlflow.set_tag("repeat", repeat)
      mlflow.log_params(hp_dict["values"])
      run_info = mlflow.active_run().info
      rid = run_info.run_id

      tb = tf.keras.callbacks.TensorBoard(
        log_dir=f"/app/logs/{ds_name}/{model_name}_{ds_id}/{rid}",
        histogram_freq=10,
        embeddings_freq=10,
        write_graph=True,
        update_freq="batch")
      stop_early = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=patience,
        restore_best_weights=True)
      model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[tb, stop_early],
        verbose=2, epochs=epochs)

      test_res = model.evaluate(test_ds, return_dict=True)
      for k, v in test_res.items():
        mlflow.log_metric(f"test_{k}", v, -1)
      return model

def evaluate(
  hypermodel_builder, fold=0, repeats=3, start_repeat=0,
  limit_id=None, **kwargs):
  if limit_id is None:
    limit_id = ed.evaluate_limit_ids[0]

  ds_id = f"{limit_id}_fold{fold}"
  dims, train_ds, val_ds, test_ds = ed.get_encoded(
    hypermodel_builder.in_enc, fold=fold, limit_id=limit_id)
  hypermodel = hypermodel_builder(**dims)

  mlflow.tensorflow.autolog(disable=True)
  tuner = em.tune_hyperparams(
    hypermodel, train_ds, val_ds, ds_id=ds_id)
  tuner.search_space_summary()

  return [
    single_evaluate(
      tuner, train_ds, val_ds, test_ds, hypermodel.name, i,
      fold=fold, limit_id=limit_id, **kwargs)
    for i in range(start_repeat, repeats)]
