import tensorflow as tf
import keras_tuner as kt

import usgoc.models.gnn as gnn

def create_model_builder(
  instanciate, with_inner_activation=False, add_hps=None):
  class HyperModel(kt.HyperModel):
    in_enc = instanciate.in_enc
    name = instanciate.name

    def __init__(self, **config):
      self.config = config

    def build(self, hp: kt.HyperParameters):
      conv_activation = hp.Choice(
        "conv_activation", ["relu", "sigmoid", "tanh", "elu"])
      if with_inner_activation:
        conv_inner_activation = hp.Choice(
          "conv_inner_activation", ["relu", "sigmoid", "tanh", "elu"])
      else:
        conv_inner_activation = conv_activation

      hp_args = dict()

      if add_hps is not None:
        hp_args = add_hps(hp)

      return instanciate(
        node_label_count=self.config["node_label_count"],
        conv_directed=True,
        conv_layer_units=[hp.Int(
          "conv_units", 32, 512, 32)] * hp.Int("conv_depth", 2, 6),
        fc_layer_units=[hp.Int(
          "fc_units", 32, 512, 32)] * hp.Int("fc_depth", 1, 3),
        conv_activation=conv_activation,
        conv_inner_activation=conv_inner_activation,
        fc_activation=hp.Choice(
          "fc_activation", ["relu", "sigmoid", "tanh", "elu"]),
        out_activation=None,
        pooling=hp.Choice(
          "pooling", ["sum", "mean", "softmax", "max", "min"]),
        learning_rate=hp.Choice("learning_rate", [1e-3, 1e-4]),
        **hp_args)

  return HyperModel

def tune_hyperparams(
  hypermodel: kt.HyperModel,
  train_ds, val_ds=None,
  max_epochs=200, patience=30,
  hyperband_iterations=3,
  overwrite=False, ds_id="") -> kt.Hyperband:
  project_name = f"{hypermodel.name}_usgo_{ds_id}"
  tuner = kt.Hyperband(
    hypermodel,
    objective="val_accuracy",
    max_epochs=max_epochs, factor=3,
    hyperband_iterations=hyperband_iterations,
    directory="/app/evaluations",
    project_name=project_name,
    overwrite=overwrite)
  stop_early = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=patience)
  tuner.search(
    train_ds, validation_data=val_ds, verbose=2,
    callbacks=[stop_early])
  return tuner

def get_best_model(tuner: kt.Tuner):
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return tuner.hypermodel.build(best_hps), best_hps.get_config()


MLPBuilder = create_model_builder(gnn.MLP)
DeepSetsBuilder = create_model_builder(gnn.DeepSets)

GCNBuilder = create_model_builder(gnn.GCN)
GINBuilder = create_model_builder(gnn.GIN)
GGNNBuilder = create_model_builder(gnn.GGNN, True)

RGCNBuilder = create_model_builder(gnn.RGCN)
RGINBuilder = create_model_builder(gnn.RGIN)