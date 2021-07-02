# import tensorflow as tf

import usgoc.datasets.unsafe_go as dataset
import usgoc.utils as utils

with utils.cache_env(use_cache=True):
  files = dataset.load_filenames()
  graphs, labels = dataset.load_dataset()

with utils.cache_env(use_cache=True):
  dims = dataset.create_dims(graphs)

dims

# files[50]
# interesting i's: 40, 70
i = 110; print(graphs[i].source_code, files[i], labels[i]); utils.draw_graph(graphs[i], edge_colors=True, layout="dot")

for a, b in graphs[i].edges(): print(graphs[i].edges.data(a, b))
