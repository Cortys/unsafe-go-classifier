import tensorflow as tf
from tensorflow import keras
import funcy as fy

import usgoc.preprocessing.tf as tf_pre
import usgoc.layers.wl1 as wl1
import usgoc.layers.pooling as pl
import usgoc.metrics.multi as mm
import usgoc.utils as utils

import warnings
warnings.filterwarnings(
  "ignore",
  "Converting sparse IndexedSlices*",
  UserWarning)

def X_normalization(input):
  X = input["X"]
  X_norm = keras.layers.BatchNormalization()(X)
  return {**input, "X": X_norm}

def pool(input, pooling="mean", preserve_full_embedding=False):
  if pooling is None:
    return input["X"]
  elif pooling == "mean":
    pool = pl.MeanPooling()
  elif pooling == "sum":
    pool = pl.SumPooling()
  elif pooling == "max":
    pool = pl.MaxPooling()
  elif pooling == "min":
    pool = pl.MinPooling()
  elif pooling == "softmax":
    pool = pl.SoftmaxPooling()
  else:
    raise AssertionError(f"Unknown pooling type '{pooling}'.")

  if preserve_full_embedding:
    return input["X"], pool(input)
  else:
    return pool(input)

def layer_stack(layer, layer_units=[], layer_args=None, **kwargs):
  layer_count = len(layer_units)
  tlayer = utils.tolerant(layer)

  def create_stack(input):
    h = input
    for i in range(layer_count):
      if layer_args is None:
        args = kwargs
      elif isinstance(layer_args, dict):
        args = fy.merge(
          kwargs, layer_args.get(i, layer_args.get(i - layer_count, {})))
      elif i < len(layer_args) and layer_args[i] is not None:
        args = fy.merge(kwargs, layer_args[i])
      else:
        args = kwargs

      units = layer_units[i]
      h = tlayer(units=units, **args)(h)
    return h

  return create_stack

def cfg_classifier(
  name, conv_layer, preproc_layer=None,
  conv_args={}, preproc_args={},
  in_enc="mwl1"):

  def loss_fn(labels, logits):
    labels = tf.squeeze(labels, -1)
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)

  label1_count = 11
  label2_count = 11

  def instanciate(
    node_label_count=None,
    conv_layer_units=[], conv_layer_args=None,
    conv_activation="sigmoid", conv_inner_activation="sigmoid",
    conv_directed=True,
    fc_layer_units=[], fc_layer_args=None,
    fc_activation="sigmoid",
    out_activation=None,
    pooling="mean",
    learning_rate=0.001):
    assert node_label_count is not None, "Missing label count."
    in_meta = dict(
      node_label_count=node_label_count,
      edge_label_count=8)

    inputs = tf_pre.make_inputs(in_enc, in_meta)

    if preproc_layer is not None:
      pre = preproc_layer(**preproc_args)(inputs)
    else:
      pre = inputs

    h = layer_stack(
      conv_layer, conv_layer_units, conv_layer_args,
      activation=conv_activation, inner_activation=conv_inner_activation,
      directed=conv_directed, **conv_args)(pre)
    pooled = pool(h, pooling)
    emb = layer_stack(
      keras.layers.Dense, fc_layer_units, fc_layer_args,
      activation=fc_activation)(pooled)

    out1 = keras.layers.Dense(
      units=label1_count, activation=out_activation,
      name="label1")(emb)
    out2 = keras.layers.Dense(
      units=label2_count, activation=out_activation,
      name="label2")(emb)
    outputs = (out1, out2)

    inputs = tf.nest.flatten(inputs)
    m = keras.Model(inputs=inputs, outputs=outputs, name=name)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = dict(label1=loss_fn, label2=loss_fn)
    acc_group = mm.SparseMultiAccuracy.create_group()
    metrics = dict(
      label1=[
        mm.SparseMultiAccuracy("label1", group=acc_group)],
      label2=[
        mm.SparseMultiAccuracy("label2", group=acc_group)])
    m.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return m

  instanciate.in_enc = in_enc
  return instanciate


# Set models:
DeepSets = cfg_classifier(
  "DeepSets", wl1.DeepSetsLayer,
  in_enc="node_set")

# Unirelational models:
GCN = cfg_classifier(
  "GCN", wl1.GCNLayer, wl1.GCNPreprocessLayer,
  in_enc="wl1")
GIN = cfg_classifier("GIN", wl1.GINLayer, in_enc="wl1")
GGNN = cfg_classifier("GGNN", wl1.GGNNLayer, in_enc="wl1")

# Multirelational models:
RGCN = cfg_classifier(
  "RGCN", wl1.RGNNLayer, wl1.RGCNPreprocessLayer,
  conv_args=dict(reverse=True),
  preproc_args=dict(reverse=True))
RGIN = cfg_classifier(
  "RGIN", wl1.RGNNLayer,
  conv_args=dict(
    reverse=True, with_final_dense=True))
