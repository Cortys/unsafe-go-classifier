import numpy as np

from usgoc.utils import tolerant
import usgoc.preprocessing.utils as enc_utils
import usgoc.preprocessing.batcher as batcher

@tolerant
def feature_dims(
  node_feature_dim=0, node_label_count=0,
  edge_feature_dim=0, edge_label_count=0,
  ignore_node_features=False, ignore_node_labels=False,
  ignore_edge_features=False, ignore_edge_labels=False,
  with_ref_features=False, multirefs=False):
  assert not with_ref_features or not multirefs,\
    "Cannot have both, continuous and discrete, edge representations."
  if ignore_node_features:
    node_feature_dim = 0
  if ignore_node_labels:
    node_label_count = 0
  if ignore_edge_features:
    edge_feature_dim = 0
  if ignore_edge_labels:
    edge_label_count = 0

  nd = max(node_feature_dim + node_label_count, 1)
  if multirefs or not with_ref_features:
    ed = 0
  else:
    ed = max(edge_feature_dim + edge_label_count, 1)

  return nd, ed

def encode_graph(
  g, node_ordering=None, edge_ordering=None,
  node_feature_dim=0, node_label_count=0,
  edge_feature_dim=0, edge_label_count=0,
  graph_feature_dim=0,
  ignore_node_features=False, ignore_node_labels=False,
  ignore_edge_features=False, ignore_edge_labels=False,
  node_label_fn=None,
  edge_label_fn=None,
  with_ref_features=False, multirefs=False, with_marked_node=False):

  if node_feature_dim is None or ignore_node_features:
    node_feature_dim = 0
  if node_label_count is None or ignore_node_labels:
    node_label_count = 0
  if edge_feature_dim is None or ignore_edge_features:
    edge_feature_dim = 0
  if edge_label_count is None or ignore_edge_labels:
    edge_label_count = 0
  if node_label_count > 0 and node_label_fn is None:
    node_label_fn = lambda d: d.get("label", 0) % node_label_count
  if edge_label_count > 0 and edge_label_fn is None:
    edge_label_fn = lambda d: d.get("label", 0) % edge_label_count

  if node_ordering is None:
    node_ordering = g.nodes

  if edge_ordering is None:
    edge_ordering = g.edges

  n_count = g.order()
  e_count = g.size()
  refs_count = max(edge_label_count, 1) if multirefs else 1
  node_dim, edge_dim = feature_dims(
    node_feature_dim, node_label_count,
    edge_feature_dim, edge_label_count,
    with_ref_features=with_ref_features, multirefs=multirefs)

  x_init = np.zeros if node_feature_dim > 0 or node_label_count > 0 \
      else np.ones

  X = x_init((n_count, node_dim), dtype=np.float32)
  if refs_count > 1:
    ref_a = tuple([] for _ in range(refs_count))
    ref_b = tuple([] for _ in range(refs_count))
    ref_sizes = np.empty((e_count + 1, refs_count), dtype=np.int32)
    ref_sizes[0, :] = 0
    ref_labels = np.empty(e_count, dtype=np.int32)
  else:
    ref_a = np.zeros(e_count, dtype=np.int32)
    ref_b = np.zeros(e_count, dtype=np.int32)
    ref_sizes = None
    ref_labels = None
  if edge_dim > 0:
    r_init = np.zeros if edge_feature_dim > 0 or edge_label_count > 0 \
        else np.ones
    R = r_init((e_count, edge_dim), dtype=np.float32)
  else:
    R = None

  n_ids = {}
  i = 0
  marked = -1 if with_marked_node else None
  for node in node_ordering:
    data = g.nodes[node]
    if node_label_count > 0:
      X[i, node_label_fn(data)] = 1
    if node_feature_dim > 0:
      X[i, node_label_count:node_dim] = data["features"]
    if with_marked_node and data.get("marked", False):
      marked = i
    n_ids[node] = i
    i += 1

  i = 0
  for e in edge_ordering:
    a = e[0]
    b = e[1]
    if edge_dim > 0:
      data = g.edges[e]
      if edge_label_count > 0:
        R[i, edge_label_fn(data)] = 1
      if edge_feature_dim > 0:
        R[i, edge_label_count:edge_dim] = data["features"]
    if refs_count == 1:
      ref_a[i] = n_ids[a]
      ref_b[i] = n_ids[b]
    else:
      j = edge_label_fn(g.edges[e])
      ref_a[j].append(n_ids[a])
      ref_b[j].append(n_ids[b])
      ref_labels[i] = j
      ref_sizes[i + 1] = ref_sizes[i]
      ref_sizes[i + 1, j] += 1
    i += 1

  if refs_count > 1:
    ref_a = tuple(np.array(r, dtype=np.int32) for r in ref_a)
    ref_b = tuple(np.array(r, dtype=np.int32) for r in ref_b)
  elif multirefs:
    ref_a = (ref_a,)
    ref_b = (ref_b,)
    ref_sizes = np.array([[i]], dtype=np.int32)

  return dict(
    X=X, R=R,
    n=n_count,
    ref_a=ref_a,
    ref_b=ref_b,
    ref_sizes=ref_sizes,
    ref_labels=ref_labels,
    marked_idx=marked)

def make_wl1_batch(
  encoded_graphs,
  flatten_multirefs=False,  # needed due to Keras nested input limitations
  discard_empty=True):
  return enc_utils.make_graph_batch(
    encoded_graphs,
    ref_keys=["ref_a", "ref_b"],
    flatten_multirefs=flatten_multirefs,
    discard_empty=discard_empty)

def make_set_batch(
  encoded_graphs,
  discard_empty=True):
  return enc_utils.make_graph_batch(
    encoded_graphs,
    ref_keys=None,
    discard_empty=discard_empty)

def vertex_count(e):
  return e["n"]

def edge_count(e):
  rs = e["ref_sizes"]
  return len(e["ref_a"]) if rs is None else np.sum(rs[-1])

def total_count(e):
  return vertex_count(e) + edge_count(e)


space_metrics = dict(
  embeddings_count=vertex_count,
  ref_count=edge_count,
  total_count=total_count
)

class WL1Batcher(batcher.Batcher):
  name = "wl1"

  def __init__(self, space_metric="embeddings_count",  **kwargs):
    super().__init__(**kwargs)
    assert space_metric in space_metrics, "Unknown WL1 space metric."
    self.space_metric = space_metric

    suffix = ""
    if self.batch_space_limit is not None:
      suffix += f"_{space_metric}_metric"

    self.name += suffix
    self.basename += suffix

  def finalize(self, graphs):
    return make_wl1_batch(graphs, flatten_multirefs=True)

  def compute_space(self, graph, batch):
    return space_metrics[self.space_metric](graph)

class SetBatcher(batcher.Batcher):
  name = "set"

  def finalize(self, graphs):
    return make_set_batch(graphs)

  def compute_space(self, graph, batch):
    return vertex_count(graph)
