import tensorflow as tf
import numpy as np
import funcy as fy
import matplotlib
import matplotlib.pyplot as plt

# import usgoc.preprocessing.graph.wl1 as wl1
import usgoc.datasets.unsafe_go as dataset
import usgoc.models.gnn as gnn
import usgoc.utils as utils

with utils.cache_env(use_cache=True):
  files = dataset.load_filenames()
  ds = dataset.load_dataset()
  graphs, targets = ds

with utils.cache_env(use_cache=True):
  # dims = dataset.create_graph_dims(graphs)
  splits = dataset.get_split_idxs(ds)
  labels1, labels2 = dataset.create_target_label_dims(ds)
  labels1 = labels1.keys()
  labels2 = labels2.keys()

# model = gnn.MLP
# model = gnn.DeepSets
# model = gnn.GCN
# model = gnn.GIN
model = gnn.GGNN
# model = gnn.RGCN
# model = gnn.RGIN

with utils.cache_env(use_cache=True):
  dims, train_ds, val_ds, test_ds = dataset.get_encoded_dataset_slices(
    ds, model.in_enc, splits, 0, limit_id="v127_d127_f127",
    batch_size_limit=200)
  train_ds = train_ds.cache()
  val_ds = val_ds.cache()

def experiment(model):
  m = model(
    node_label_count=dims["node_label_count"],
    conv_directed=True,
    conv_layer_units=[64] * 5, fc_layer_units=[64] * 5,
    conv_activation="relu",
    conv_inner_activation="relu",
    fc_activation="relu",
    out_activation=None,
    pooling="sum", learning_rate=0.001)

  m.fit(train_ds, validation_data=val_ds, verbose=2, epochs=500)
  res = m.evaluate(test_ds, return_dict=True)
  print(res)
  return m


m = experiment(model)

# interesting i's: 40, 70, 874 (44b41ab329d2624a449e)
j = 6
i = splits[0]["test"][j]
# i = fy.first(fy.filter(lambda e: "e6a3feac3fc40c305816" in e[1], enumerate(files)))[0]
i, files[i]
# graphs[i]
i = i; print(graphs[i].source_code, files[i], targets[0][i], targets[1][i]); utils.draw_graph(graphs[i], edge_colors=True, layout="dot")

with utils.cache_env(use_cache=False):
  singleton_ds = dataset.dataset_encoders[model.in_enc](dataset.slice(ds, [i]), dims)
list(singleton_ds)
s_pred = m.predict(test_ds)
# s_pred = m.predict(singleton_ds)
np.around(tf.nn.softmax(s_pred[0], -1).numpy()[j], 3)
np.around(tf.nn.softmax(s_pred[1], -1).numpy()[j], 3)
fy.zipdict(labels1, list(s_pred[0][j]))
fy.zipdict(labels2, list(s_pred[1][j]))
s_pred_labels = tf.cast(tf.stack([tf.argmax(s_pred[0], -1), tf.argmax(s_pred[1], -1)], 1), tf.int32)
s_pred_labels[j]
target_labels = tf.stack(list(test_ds)[0][1], 1)
s_pred_labels == target_labels
