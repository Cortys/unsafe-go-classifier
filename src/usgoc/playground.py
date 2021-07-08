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

# model = gnn.DeepSets
model = gnn.GCN
# model = gnn.GIN
# model = gnn.GGNN
# model = gnn.RGCN
# model = gnn.RGIN

with utils.cache_env(use_cache=True):
  dims, train_ds, val_ds, test_ds = dataset.get_encoded_dataset_slices(
    ds, model.in_enc, splits, 0, limit_id="v127_d127_f127",
    batch_size_limit=200)
  train_ds = train_ds.cache()
  val_ds = val_ds.cache()

# list(test_ds)[0]
m = model(
  node_label_count=dims["node_label_count"],
  conv_directed=True,
  conv_layer_units=[64] * 3, fc_layer_units=[64] * 2,
  conv_activation="relu",
  conv_inner_activation="relu",
  fc_activation="relu",
  out_activation=None,
  pooling="sum", learning_rate=0.001)


m.fit(train_ds, validation_data=val_ds, verbose=2, epochs=500)
res = m.evaluate(test_ds, return_dict=True)
print(res)

# interesting i's: 40, 70, 874 (44b41ab329d2624a449e)
i = fy.first(fy.filter(lambda e: "44b41ab329d2624a449e" in e[1], enumerate(files)))[0]
i, files[i]
i
i = i; print(graphs[i].source_code, files[i], targets[0][i], targets[1][i]); utils.draw_graph(graphs[i], edge_colors=True, layout="dot")

with utils.cache_env(use_cache=False):
  singleton_ds = dataset.dataset_encoders[model.in_enc](dataset.slice(ds, [i]), dims)

singleton_ds
s_pred = m.predict(singleton_ds)
s_pred[0]
tf.argmax(s_pred[0], -1), tf.argmax(s_pred[1], -1)
