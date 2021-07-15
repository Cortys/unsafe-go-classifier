import tensorflow as tf
import mlflow

import usgoc.evaluation.models as em
import usgoc.evaluation.datasets as ed

mlflow.set_tracking_uri("file://app/mlruns")
mlflow.keras.autolog()

def evaluate(
  hypermodel_builder, fold=0,
  epochs=1000, patience=100,
  limit_id=None):
  if limit_id is None:
    limit_id = ed.evaluate_limit_ids[0]

  ds_id = f"{limit_id}_fold{fold}"

  dims, train_ds, val_ds, test_ds = ed.get_encoded(
    hypermodel_builder.in_enc, fold=fold, limit_id=limit_id)
  hypermodel = hypermodel_builder(**dims)

  tuner = em.tune_hyperparams(
    hypermodel, train_ds, val_ds, ds_id=ds_id)
  tuner.search_space_summary()

  with mlflow.start_run():
    model, hp_dict = em.get_best_model(tuner)
    mlflow.set_tag("model", hypermodel.name)
    mlflow.set_tag("limit_id", limit_id)
    mlflow.set_tag("fold", fold)
    mlflow.log_params(hp_dict)

    tb = tf.keras.callbacks.Tensorboard(
      log_dir=f"/app/logs/{hypermodel.name}_{ds_id}",
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
      verbose=2, epochs=1000)

    return model
