# import tensorflow as tf

import usgoc.datasets.unsafe_go as dataset
import usgoc.utils as utils

with utils.cache_env(use_cache=False):
  files = dataset.load_filenames()
  graphs, labels = dataset.load_dataset()

files[40]
i = 40; print(graphs[i].source_code, files[i], labels[i]); utils.draw_graph(graphs[i], edge_colors=True, layout="dot")
