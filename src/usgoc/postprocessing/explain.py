import funcy as fy
import tensorflow as tf
import numpy as np

def enrich_model_outputs(model, with_conv_output=False):
  padded_X_layer = fy.first(
    fy.filter(lambda l: "tensor_scatter" in l.name, model.layers))
  pooling_layer = fy.first(
    fy.filter(lambda l: "pooling" in l.name, model.layers))

  if padded_X_layer is not None and pooling_layer is not None:
    if with_conv_output:
      additional_outputs = (padded_X_layer.output, pooling_layer.input["X"])
    else:
      additional_outputs = (padded_X_layer.output,)
  else:
    marked_X_layer = fy.first(
        fy.filter(lambda l: "gather" in l.name, model.layers))
    additional_outputs = (marked_X_layer.output,)
  return tf.keras.Model(model.input, outputs=model.output + additional_outputs)

def compute_batch_importances(model, input):
  X = input["X"]
  N = input["n"].shape[0]
  graph_X = input["graph_X"]
  class_counts = (
      model.output[0].shape[-1],
      model.output[1].shape[-1])
  vertex_importance = ([], [])
  vertex_feature_importance = ([], [])
  graph_feature_importance = ([], [])

  with tf.GradientTape(persistent=True) as g:
    g.watch(X)
    g.watch(graph_X)
    output = model(input)
    if len(output) == 4:
      padded_X, conv_X = output[-2:]
    elif len(output) == 3:
      conv_X = None
      padded_X = output[-1]

    for label_idx in [0, 1]:
      for c in range(class_counts[label_idx]):
        pred = output[label_idx][:, c]
        with g.stop_recording():
          if conv_X is not None:
            X_grad = g.gradient(pred, conv_X)
            vertex_importance[label_idx].append(tf.nn.relu(
                tf.math.reduce_sum(X_grad * conv_X, axis=-1)))
          X_grad, graph_grad = g.gradient(pred, (padded_X, graph_X))
          fimp = X_grad * padded_X
          if padded_X.shape[0] != N:
            fimp = tf.math.unsorted_segment_sum(
                fimp, input["graph_idx"], N)
          vertex_feature_importance[label_idx].append(fimp)
          graph_feature_importance[label_idx].append(graph_grad * graph_X)

  if conv_X is None:
    return vertex_feature_importance, graph_feature_importance
  else:
    return vertex_importance, vertex_feature_importance, graph_feature_importance

def compute_importances(model, ds):
  model = enrich_model_outputs(model, with_conv_output=False)
  l1_fimps, l2_fimps = [], []

  for input, _ in ds:
    (l1_fimp, l2_fimp), (l1_gimp, l2_gimp) = compute_batch_importances(model, input)
    l1_fimps.append(tf.concat([l1_fimp, l1_gimp], axis=-1))
    l2_fimps.append(tf.concat([l2_fimp, l2_gimp], axis=-1))

  l1_fimps = tf.concat(l1_fimps, axis=1).numpy()
  l2_fimps = tf.concat(l2_fimps, axis=1).numpy()
  return l1_fimps, l2_fimps

def group_feature_importance(fimps):
  fimps_instance_grp = np.sum(fimps, 1)
  fimps_lbl_grp = np.sum(fimps_instance_grp, 0, keepdims=True)
  return np.concatenate(
    [fimps_instance_grp, fimps_lbl_grp],
    axis=0)
