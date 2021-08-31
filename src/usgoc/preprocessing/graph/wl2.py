import numpy as np
import networkx as nx

from usgoc.utils import tolerant
import usgoc.preprocessing.utils as enc_utils
import usgoc.preprocessing.batcher as batcher

@tolerant
def feature_dim(
  node_feature_dim=0, node_label_count=0,
  edge_feature_dim=0, edge_label_count=0,
  ignore_node_features=False, ignore_node_labels=False,
  ignore_edge_features=False, ignore_edge_labels=False):
  if ignore_node_features:
    node_feature_dim = 0
  if ignore_node_labels:
    node_label_count = 0
  if ignore_edge_features:
    edge_feature_dim = 0
  if ignore_edge_labels:
    edge_label_count = 0

  return 3 + node_feature_dim + node_label_count \
      + edge_feature_dim + edge_label_count

def eid_lookup(e_ids, i, j):
  if i > j:
    i, j = j, i

  return e_ids[(i, j)]

def encode_graph(
  g, radius=1,
  node_feature_dim=0, node_label_count=0,
  edge_feature_dim=0, edge_label_count=0,
  graph_feature_dim=0,
  ignore_node_features=False, ignore_node_labels=False,
  ignore_edge_features=False, ignore_edge_labels=False,
  node_label_fn=None,
  edge_label_fn=None,
  with_marked_node=False):

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

  marked = -1 if with_marked_node else None
  multi = isinstance(g, nx.MultiGraph)
  node_label_dims = dict()

  if node_label_count > 0:
    remove_nodes = set()
    for n in g.nodes:
      d = g.nodes[n]
      label_dims = node_label_fn(g.nodes[n])
      if label_dims is False:
        remove_nodes.add(n)
      node_label_dims[n] = label_dims
    if len(remove_nodes) > 0:
      g = g.copy()
      g.remove_nodes_from(remove_nodes)

  if isinstance(g, nx.DiGraph):
    g_base = g.to_undirected(as_view=True)
    g_p = nx.power(nx.Graph(g), radius)
  elif multi:
    g_base = g
    g_p = nx.power(nx.Graph(g), radius)
  else:
    g_base = g
    g_p = nx.power(g_base, radius)

  for node in g.nodes:
    g_p.add_edge(node, node)

  e_count = g_p.size()
  X_dim = feature_dim(
    node_feature_dim, node_label_count,
    edge_feature_dim, edge_label_count)
  X = np.zeros((e_count, X_dim), dtype=np.float32)
  ref_a = []
  ref_b = []
  backref = []

  node_label_offset = 3
  node_feature_offset = node_label_offset + node_label_count
  edge_label_offset = node_feature_offset + node_feature_dim
  edge_feature_offset = edge_label_offset + edge_label_count
  e_ids = {e: i for i, e in enumerate(g_p.edges)}
  i = 0
  for edge in g_p.edges:
    a, b = edge
    neighbors = list(nx.common_neighbors(g_p, a, b))
    n_count = len(neighbors)
    n_a = [eid_lookup(e_ids, a, k) for k in neighbors]
    n_b = [eid_lookup(e_ids, b, k) for k in neighbors]

    if a == b:
      X[i, 0] = 1
      if node_label_count > 0:
        label_dims = node_label_dims[a]
        X[i, node_label_offset + label_dims] = 1
      if node_feature_dim > 0:
        d = g_base.nodes[a]
        X[i, node_feature_offset:edge_label_offset] = d["features"]
      if with_marked_node and g_base.nodes[a].get("marked", False):
        marked = i

      ref_a.append(i)
      ref_b.append(i)
      n_count += 1
    else:
      if g_base.has_edge(a, b):
        X[i, 1] = 1
        d: dict = g_base.get_edge_data(a, b)
        if multi:
          if edge_label_count > 0:
            for sd in d.values():
              X[i, edge_label_offset + edge_label_fn(sd)] = 1
          if edge_feature_dim > 0:
            for sd in d.values():
              X[i, edge_feature_offset:X_dim] += sd["features"]
        else:
          if edge_label_count > 0:
            X[i, edge_label_offset + edge_label_fn(d)] = 1
          if edge_feature_dim > 0:
            X[i, edge_feature_offset:X_dim] = d["features"]
      else:
        X[i, 2] = 1
      ref_a += [i, eid_lookup(e_ids, a, a)]
      ref_b += [eid_lookup(e_ids, b, b), i]
      n_count += 2

    ref_a += n_a
    ref_b += n_b
    backref += [i] * n_count
    i += 1

  return dict(
    X=X,
    ref_a=np.array(ref_a),
    ref_b=np.array(ref_b),
    backref=np.array(backref),
    n=g.order(),
    marked_idx=marked)

def make_wl2_batch(
  encoded_graphs,
  discard_empty=True):
  return enc_utils.make_graph_batch(
    encoded_graphs,
    ref_keys=["ref_a", "ref_b", "backref"],
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

def ref_count(e):
  return len(e["ref_a"])

def embeddings_count(e):
  return len(e["X"])

def total_count(e):
  return len(e["X"]) + len(e["ref_a"])


space_metrics = dict(
  vertex_count=vertex_count,
  embeddings_count=embeddings_count,
  ref_count=ref_count,
  total_count=total_count
)

class WL2Batcher(batcher.Batcher):
  name = "wl2"

  def __init__(self, space_metric="embeddings_count",  **kwargs):
    super().__init__(**kwargs)
    assert space_metric in space_metrics, "Unknown WL2 space metric."
    self.space_metric = space_metric

    suffix = ""
    if self.batch_space_limit is not None:
      suffix += f"_{space_metric}_metric"

    self.name += suffix
    self.basename += suffix

  def finalize(self, graphs):
    return make_wl2_batch(graphs)

  def compute_space(self, graph, batch):
    return space_metrics[self.space_metric](graph)
