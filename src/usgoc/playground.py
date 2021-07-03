# import tensorflow as tf
import numpy as np

import usgoc.preprocessing.graph.wl1 as wl1
import usgoc.datasets.unsafe_go as dataset
import usgoc.utils as utils

with utils.cache_env(use_cache=True):
  files = dataset.load_filenames()
  graphs, labels = dataset.load_dataset()

with utils.cache_env(use_cache=True):
  dims = dataset.create_dims(graphs)

dims

with utils.cache_env(use_cache=False):
  e = dataset.wl1_tf_dataset((graphs, labels), dims, batch_size_limit=10)

list(e)[0]

# files[50]
# interesting i's: 40, 70
# i = 110; print(graphs[i].source_code, files[i], labels[i]); utils.draw_graph(graphs[i], edge_colors=True, layout="dot")
