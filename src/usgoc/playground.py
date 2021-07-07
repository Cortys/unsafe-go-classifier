import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# import usgoc.preprocessing.graph.wl1 as wl1
import usgoc.datasets.unsafe_go as dataset
import usgoc.models.gnn as gnn
import usgoc.utils as utils

with utils.cache_env(use_cache=True):
  files = dataset.load_filenames()
  ds = dataset.load_dataset()

with utils.cache_env(use_cache=True):
  # dims = dataset.create_graph_dims(graphs)
  splits = dataset.get_split_idxs(ds)
  labels1, labels2 = dataset.create_target_label_dims(ds)
  labels1 = labels1.keys()
  labels2 = labels2.keys()

with utils.cache_env(use_cache=True):
  dims, train_ds, val_ds, test_ds = dataset.wl1_tf_datasets(
    ds, splits, 0, batch_size_limit=200)
  train_ds = train_ds.cache()
  val_ds = val_ds.cache()

m = gnn.RGCN(
  node_label_count=dims["node_label_count"],
  conv_layer_units=[256] * 5, fc_layer_units=[128] * 2,
  conv_activation="relu",
  fc_activation="relu",
  out_activation=None,
  pooling="sum", learning_rate=0.001)

m.fit(train_ds, validation_data=val_ds, verbose=2, epochs=500)
res = m.evaluate(test_ds, return_dict=True)
print(res)


# files[50]
# interesting i's: 40, 70
# i = 110; print(graphs[i].source_code, files[i], labels[i]); utils.draw_graph(graphs[i], edge_colors=True, layout="dot")
