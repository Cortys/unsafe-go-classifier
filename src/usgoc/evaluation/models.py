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
        "conv_activation", ["relu", "sigmoid", "tanh"])
      if with_inner_activation:
        conv_inner_activation = hp.Choice(
          "conv_inner_activation", ["relu", "sigmoid", "tanh"])
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
          "fc_activation", ["relu", "sigmoid", "tanh"]),
        out_activation=None,
        pooling=hp.Choice(
          "pooling", ["sum", "mean", "softmax", "max"]),
        learning_rate=hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4]),
        **hp_args)

  return HyperModel


MLPBuilder = create_model_builder(gnn.MLP)
DeepSetsBuilder = create_model_builder(gnn.DeepSets)

GCNBuilder = create_model_builder(gnn.GCN)
GINBuilder = create_model_builder(gnn.GIN)
GGNNBuilder = create_model_builder(gnn.GGNN, True)

RGCNBuilder = create_model_builder(gnn.RGCN)
RGINBuilder = create_model_builder(gnn.RGIN)
