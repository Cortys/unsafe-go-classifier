import glob
import json
from pathlib import Path
from collections import defaultdict
import funcy as fy
import numpy as np

import usgoc.utils as utils
import usgoc.preprocessing.graph.wl1 as wl1
import usgoc.preprocessing.transformer as trans
import usgoc.preprocessing.batcher as batcher
import usgoc.preprocessing.tf as tf
import usgoc.preprocessing.split as ps
import usgoc.datasets.go_cfg_utils as cfg_utils

RAW_DATASET_PATTERN = "/app/raw/unsafe-go-dataset/**/*.json"
DATA_DIR = Path("/app/data/unsafe-go-dataset")

no_default_label_subtype = {
  "type", "blocktype", "vartype"
}
node_label_type_dim_count = {
  "v0_d0_f0_p0": dict(
    varname=0,
    datatype=0,
    function=0,
    package=0
  ),
  "v0_d63_f63_p63": dict(
    varname=0,
    datatype=63,
    function=63,
    package=63
  ),
  "v15_d63_f63_p63": dict(
    varname=15,
    datatype=63,
    function=63,
    package=63
  ),
  "v63_d63_f63_p63": dict(
    varname=63,
    datatype=63,
    function=63,
    package=63
  ),
  "v127_d127_f127_p127": dict(
    varname=127,
    datatype=127,
    function=127,
    package=127
  ),
  "v255_d255_f255_p255": dict(
    varname=255,
    datatype=255,
    function=255,
    package=255
  )
}
default_limit_id = "v127_d127_f127_p127"
graph_features = dict()
graph_feature_eye = np.eye(len(cfg_utils.cfg_types), dtype=np.float32)
for i, t in enumerate(cfg_utils.cfg_types):
  graph_features[t] = graph_feature_eye[i]

def load_filenames():
  return glob.glob(RAW_DATASET_PATTERN, recursive=True)

def load_raw():
  files = load_filenames()
  res = []
  for f in files:
    with open(f, "r") as fp:
      res.append(json.load(fp))
  return res

@utils.cached(DATA_DIR, "cfgs")
def raw_to_graphs(raw_dataset):
  graphs = np.empty(len(raw_dataset), dtype="O")
  for i, inst in enumerate(raw_dataset):
    graphs[i] = cfg_utils.cfg_to_graph(
      inst["cfg"], int(inst["usage"]["line"]))

  return graphs

@utils.cached(DATA_DIR, "target_label_dims", "pretty_json")
def create_target_label_dims(raw_dataset):
  label1s = set()
  label2s = set()
  for inst in raw_dataset:
    usage = inst["usage"]
    label1s.add(usage["label1"])
    label2s.add(usage["label2"])
  label1s = sorted(label1s)
  label2s = sorted(label2s)
  label1s = dict(zip(label1s, range(len(label1s))))
  label2s = dict(zip(label2s, range(len(label2s))))
  return label1s, label2s

@utils.cached(DATA_DIR, "target_labels")
def raw_to_usages(raw_dataset):
  label1s, label2s = create_target_label_dims(raw_dataset)
  usage1s = np.empty(len(raw_dataset), dtype=np.int32)
  usage2s = np.empty(len(raw_dataset), dtype=np.int32)
  for i, inst in enumerate(raw_dataset):
    usage = inst["usage"]
    usage1s[i] = label1s[usage["label1"]]
    usage2s[i] = label2s[usage["label2"]]
  return usage1s, usage2s

def load_dataset():
  raw_dataset = load_raw()
  return raw_to_graphs(raw_dataset), raw_to_usages(raw_dataset)

@utils.cached(
  DATA_DIR / "node_label_histogram",
  lambda _, split_id=None: (
    f"node_labels{'' if split_id is None else '_' + split_id}"),
  "pretty_json")
def collect_node_label_histogram(graphs, split_id=None):
  labels = {
    type: defaultdict(lambda: 0)
    for type in cfg_utils.node_label_types.keys()}

  for g in graphs:
    for v, data in g.nodes(data=True):
      v_labels = data["labels"]
      for lt, ls in v_labels:
        labels[lt][ls] += 1

  for lt, ls in labels.items():
    ls = sorted(ls.items(), key=lambda e: e[1], reverse=True)
    labels[lt] = ls

  return labels

@utils.cached(
  DATA_DIR / "cfg_dims",
  lambda _, limit_id=default_limit_id, split_id=None: (
    f"dims_{limit_id}{'' if split_id is None else '_' + split_id}"),
  "pretty_json")
def create_graph_dims(graphs, limit_id=default_limit_id, split_id=None):
  labels = collect_node_label_histogram(graphs, split_id)
  node_dim_count = 0
  node_dims = dict()
  edge_dims = dict()
  dim_limits = node_label_type_dim_count.get(limit_id, {})

  for lt in cfg_utils.node_label_types.keys():
    ls = labels[lt]
    d = dict()
    node_dims[lt] = d
    if lt not in no_default_label_subtype:
      d[""] = node_dim_count
      node_dim_count += 1

    limit = dim_limits.get(lt, None)
    if limit is not None:
      ls = ls[:limit]
    for lb, c in ls:
      d[lb] = node_dim_count
      node_dim_count += 1

  for i, ls in enumerate(cfg_utils.edge_labels):
    edge_dims[ls] = i

  return dict(
    node_labels=node_dims,
    node_label_count=node_dim_count,
    edge_labels=edge_dims,
    edge_label_count=len(cfg_utils.edge_labels))

def get_node_label_dims(node_dims, node_data):
  labels = node_data["labels"]

  def lookup(label):
    lt, ls = label
    d = node_dims[lt]
    if ls in d:
      return d[ls]
    return d[""]

  return fy.lmap(lookup, labels)

def get_edge_label_dim(edge_dims, edge_data):
  return edge_dims[edge_data["label"]]

@utils.cached(
  DATA_DIR / "wl1",
  lambda graphs, dims, multirefs=True, split_id=None: (
    ("m" if multirefs else "")
    + "wl1"
    + ("" if split_id is None else "_" + split_id)))
def wl1_encode_graphs(graphs, dims, multirefs=True, split_id=None):
  enc_graphs = np.empty(len(graphs), dtype="O")
  node_label_fn = fy.partial(get_node_label_dims, dims["node_labels"])
  edge_label_fn = fy.partial(get_edge_label_dim, dims["edge_labels"])
  node_label_count = dims["node_label_count"]
  edge_label_count = dims["edge_label_count"]

  for i, g in enumerate(graphs):
    enc_g = wl1.encode_graph(
      g,
      node_label_count=node_label_count,
      edge_label_count=edge_label_count,
      node_label_fn=node_label_fn,
      edge_label_fn=edge_label_fn,
      multirefs=multirefs,
      with_marked_node=True)
    enc_g["graph_X"] = graph_features[g.cfg_type]
    enc_graphs[i] = enc_g

  return enc_graphs

def wl1_tf_dataset(
  dataset, dims, multirefs=True, split_id=None,
  batch_size_limit=None, batch_space_limit=None):
  graphs, targets = dataset
  encoded_graphs = wl1_encode_graphs(graphs, dims, multirefs, split_id)

  node_label_count = dims["node_label_count"]
  edge_label_count = dims["edge_label_count"]
  ds_batcher = trans.tuple(
    wl1.WL1Batcher(
      batch_size_limit=batch_size_limit,
      batch_space_limit=batch_space_limit),
    trans.pair(batcher.Batcher.identity))
  gen = ds_batcher.batch_generator((encoded_graphs, targets))

  in_enc = "mwl1" if multirefs else "wl1"
  in_meta = dict(
    node_label_count=node_label_count,
    edge_label_count=edge_label_count,
    graph_feature_dim=len(graph_features),
    with_marked_node=True)
  out_meta = dict()
  return tf.make_dataset(gen, in_enc, in_meta, "int32_pair", out_meta)

def node_set_tf_dataset(
  dataset, dims, split_id=None,
  batch_size_limit=None, batch_space_limit=None):
  graphs, targets = dataset
  encoded_graphs = wl1_encode_graphs(graphs, dims, True, split_id)

  node_label_count = dims["node_label_count"]
  ds_batcher = trans.tuple(
    wl1.SetBatcher(
      batch_size_limit=batch_size_limit,
      batch_space_limit=batch_space_limit),
    trans.pair(batcher.Batcher.identity))
  gen = ds_batcher.batch_generator((encoded_graphs, targets))

  in_meta = dict(
    node_label_count=node_label_count,
    graph_feature_dim=len(graph_features),
    with_marked_node=True)
  out_meta = dict()
  return tf.make_dataset(gen, "node_set", in_meta, "int32_pair", out_meta)

def slice(dataset, indices=None):
  if indices is None:
    return dataset

  graphs, labels = dataset
  label1s, label2s = labels
  graphs = graphs[indices]
  label1s = label1s[indices]
  label2s = label2s[indices]

  return graphs, (label1s, label2s)

@utils.cached(DATA_DIR, "splits", "json")
def get_split_idxs(dataset):
  graphs, labels = dataset
  n = len(graphs)
  # Assign unique class to each label combination:
  # strat_labels = labels[0]
  strat_labels = labels[0] * 12 + labels[1] + 1
  # Stratify rare combinations by their first label only:
  bins = np.bincount(strat_labels)
  unique_combinations = np.nonzero((0 < bins) & (bins < 10))[0]
  rare_idxs = np.isin(strat_labels, unique_combinations)
  strat_labels[rare_idxs] = 12 * (strat_labels[rare_idxs] // 12)

  return ps.make_splits(n, strat_labels=strat_labels)

def get_dataset_slices(dataset, split_idxs, outer_i=0, inner_i=0):
  split = split_idxs[outer_i]
  ms = split["model_selection"][inner_i]
  train_idxs = ms["train"]
  val_idxs = ms["validation"]
  test_idxs = split["test"]
  train_slice = slice(dataset, train_idxs)
  val_slice = slice(dataset, val_idxs)
  test_slice = slice(dataset, test_idxs)
  return train_slice, val_slice, test_slice


dataset_encoders = dict(
  wl1=fy.partial(wl1_tf_dataset, multirefs=False),
  mwl1=wl1_tf_dataset,
  node_set=node_set_tf_dataset
)

def get_encoded_dataset_slices(
  dataset, enc, split_idxs,
  outer_i=0, inner_i=0,
  limit_id=default_limit_id, **kwargs):
  split_id = f"{outer_i}_{inner_i}"
  encoder = dataset_encoders[enc]
  train_slice, val_slice, test_slice = get_dataset_slices(
    dataset, split_idxs, outer_i, inner_i)
  train_dims = create_graph_dims(
    train_slice[0], limit_id, f"{split_id}_train")

  train_ds = encoder(
    train_slice, train_dims, split_id=f"{limit_id}_{split_id}_train", **kwargs)
  val_ds = encoder(
    val_slice, train_dims, split_id=f"{limit_id}_{split_id}_val", **kwargs)
  test_ds = encoder(
    test_slice, train_dims, split_id=f"{limit_id}_{split_id}_test", **kwargs)
  return train_dims, train_ds, val_ds, test_ds
