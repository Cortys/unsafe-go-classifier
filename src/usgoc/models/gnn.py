import tensorflow as tf
from tensorflow import keras
import funcy as fy

import usgoc.preprocessing.tf as tf_pre
import usgoc.layers.wl1 as wl1
import usgoc.layers.wl2 as wl2
import usgoc.layers.pooling as pl
from usgoc.layers.utils import with_reg, conv_with_reg
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

def pool(input, pooling="mean", preserve_unpooled=False):
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

  if preserve_unpooled:
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
  name, conv_layer=None, preproc_layer=None,
  conv_args={},
  in_enc="mwl1"):

  label1_count = 11
  label2_count = 11

  def instanciate(
    node_label_count=None,
    fc_layer_units=[], fc_layer_args=None,
    out_activation=None,
    pooling="mean",
    learning_rate=0.001,
    **kwargs):
    assert node_label_count is not None, "Missing label count."

    conv_args_ext = utils.select_prefixed_keys(
      kwargs, "conv_", True, conv_args.copy())
    fc_args_ext = utils.select_prefixed_keys(
      kwargs, "fc_", True)

    in_meta = dict(
      node_label_count=node_label_count,
      graph_feature_dim=4,
      edge_label_count=9,
      with_marked_node=True)

    inputs = tf_pre.make_inputs(in_enc, in_meta)
    marked_idx = tf.reshape(inputs["marked_idx"], [-1])
    graph_X = inputs["graph_X"]

    if conv_layer is None:
      marked_X = tf.gather(inputs["X"], marked_idx, axis=0)
      combined_X = tf.concat([graph_X, marked_X], 1)
    else:
      batch_size = tf.shape(marked_idx)[0]
      padded_X = tf.pad(inputs["X"], tf.constant([[0, 0], [1, 0]]))
      padded_X = tf.tensor_scatter_nd_update(
        padded_X,
        tf.stack([marked_idx, tf.zeros(batch_size, dtype=tf.int32)], axis=1),
        tf.ones(batch_size))
      padded_input = {**inputs, "X": padded_X}

      if preproc_layer is not None:
        padded_input = utils.tolerant(preproc_layer)(
          **conv_args_ext)(padded_input)

      h = layer_stack(
        conv_with_reg(conv_layer),
        **conv_args_ext)(padded_input)
      X, pooled_X = pool(h, pooling, preserve_unpooled=True)
      marked_X = tf.gather(X, marked_idx, axis=0)
      combined_X = tf.concat([graph_X, marked_X, pooled_X], 1)

    emb = layer_stack(
      with_reg(keras.layers.Dense),
      fc_layer_units, fc_layer_args,
      **fc_args_ext)(combined_X)

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
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc_group = mm.SparseMultiAccuracy.create_group()
    metrics = dict(
      label1=[
        mm.SparseMultiAccuracy("label1", group=acc_group)],
      label2=[
        mm.SparseMultiAccuracy("label2", group=acc_group)])
    m.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return m

  instanciate.in_enc = in_enc
  instanciate.name = name
  return instanciate


# Set models:
MLP = cfg_classifier("MLP", in_enc="node_set")
DeepSets = cfg_classifier(
  "DeepSets", wl1.DeepSetsLayer,
  in_enc="node_set")

# Unirelational models:
GCN = cfg_classifier(
  "GCN", wl1.GCNLayer, wl1.GCNPreprocessLayer,
  in_enc="wl1")
GIN = cfg_classifier("GIN", wl1.GINLayer, in_enc="wl1")
GGNN = cfg_classifier(
  "GGNN", wl1.GGNNLayer,
  conv_args=dict(use_diff=True, reverse=True),
  in_enc="wl1")
WL2GNN = cfg_classifier("WL2GNN", wl2.WL2Layer, in_enc="wl2")

# Multirelational models:
RGCN = cfg_classifier(
  "RGCN", wl1.RGNNLayer, wl1.RGCNPreprocessLayer,
  conv_args=dict(reverse=True))
RGIN = cfg_classifier(
  "RGIN", wl1.RGNNLayer,
  conv_args=dict(
    reverse=True, with_final_dense=True))
