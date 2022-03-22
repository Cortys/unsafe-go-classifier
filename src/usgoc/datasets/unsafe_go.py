import glob
import json
from pathlib import Path
from collections import defaultdict
import funcy as fy
import numpy as np

import usgoc.utils as utils
import usgoc.preprocessing.graph.wl1 as wl1
import usgoc.preprocessing.graph.wl2 as wl2
import usgoc.preprocessing.transformer as trans
import usgoc.preprocessing.batcher as batcher
import usgoc.preprocessing.tf as tf
import usgoc.preprocessing.split as ps
import usgoc.datasets.go_cfg_utils as cfg_utils

RAW_DATASET_PATTERN = f"{utils.PROJECT_ROOT}/raw/unsafe-go-dataset/**/*.json"
DATA_DIR = Path(f"{utils.PROJECT_ROOT}/data/unsafe-go-dataset")

no_default_label_subtype = {
  "type", "blocktype", "vartype"
}

def get_limit_name(limit_dict):
  if isinstance(limit_dict, str):
    return limit_dict

  v = limit_dict["varname"]
  d = limit_dict["datatype"]
  f = limit_dict["function"]
  p = limit_dict["package"]
  if v == cfg_utils.is_semantic_name:
    v = "S"

  excl_suffix = ""
  only_suffix = ""
  types: set = limit_dict["type"]

  if "block" not in types:
    types.remove("subblock")
    excl_suffix += "_b"
    f = 0
    limit_dict["function"] = 0
  if "subblock" not in types:
    excl_suffix = "_sb"
  if "var" not in types:
    excl_suffix += "_v"
    v = 0
    limit_dict["varname"] = 0
    limit_dict["vartype"] = False

  if not limit_dict["vartype"]:
    excl_suffix += "_vt"
  if not limit_dict["blocktype"]:
    excl_suffix += "_bt"
  if not limit_dict["datatype_flag"]:
    excl_suffix += "_tf"
  elif isinstance(limit_dict["datatype_flag"], set):
    only_suffix = "_tf" + "".join(sorted(limit_dict["datatype_flag"]))
  if not limit_dict["builtin_function"]:
    excl_suffix += "_fb"
  if not limit_dict["binary_op"]:
    excl_suffix += "_ob"
  if not limit_dict["unary_op"]:
    excl_suffix += "_ou"
  if not limit_dict["selfref"]:
    excl_suffix += "_s"

  if d == 0 and f == 0 and p == 0:
    limit_dict["only_core_packages"] = False

  limit_name = f"v{v}_d{d}_f{f}_p{p}"
  if limit_dict["only_core_packages"]:
    limit_name += "_core"
  if only_suffix != "":
    limit_name += f"_only{only_suffix}"
  if excl_suffix != "":
    limit_name += f"_no{excl_suffix}"

  return limit_name

@utils.memoize
def get_dim_limit_dict():
  dim_limits = dict(all={})
  limit_combinations = utils.cart(
    varname=[0, 127, cfg_utils.is_semantic_name],
    type=[{"block", "subblock", "var"}, {"block", "subblock"}],
    datatype=[0, 127],
    function=[0, 127],
    package=[0, 127],
    blocktype=[False, True],
    selfref=[False, True],
    vartype=[False, True],
    datatype_flag=[False, True],
    builtin_function=[False, True],
    binary_op=[False, True],
    unary_op=[False, True],
    only_core_packages=[False, True],
  )

  for limit_dict in limit_combinations:
    limit_name = get_limit_name(limit_dict)
    dim_limits[limit_name] = limit_dict
  return dim_limits


convert_modes = cfg_utils.convert_modes
default_mode = "atomic_blocks"
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

@utils.cached(DATA_DIR, lambda _, mode=default_mode: f"cfgs_{mode}")
def raw_to_graphs(raw_dataset, mode=default_mode):
  graphs = np.empty(len(raw_dataset), dtype="O")
  for i, inst in enumerate(raw_dataset):
    if i % 100 == 99:
      print(f"[dbg] Preprocessing CFG {i+1}/{len(raw_dataset)} (mode={mode}).")
    graphs[i] = cfg_utils.cfg_to_graph(
      inst["cfg"],
      int(inst["usage"]["line"]),
      inst["usage"]["module"],
      mode=mode)

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

def load_dataset(mode=default_mode):
  raw_dataset = load_raw()
  return raw_to_graphs(raw_dataset, mode), raw_to_usages(raw_dataset)

@utils.cached(
  DATA_DIR / "node_label_histogram",
  lambda _, split_id=None, mode=default_mode: (
    f"node_labels_{mode}{'' if split_id is None else '_' + split_id}"),
  "pretty_json")
def collect_node_label_histogram(graphs, split_id=None, mode=default_mode):
  labels = {
    type: defaultdict(lambda: 0)
    for type in cfg_utils.node_label_types.keys()}
  is_core_package = dict()
  is_core_datatype = dict()
  is_core_function = dict()

  for g in graphs:
    types_to_pkgs = g.types_to_pkgs
    funcs_to_pkgs = g.funcs_to_pkgs
    for v, data in g.nodes(data=True):
      v_labels = data["labels"]
      for lt, ls in v_labels:
        labels[lt][ls] += 1
        if lt == "package" and ls not in is_core_package:
          is_core_package[ls] = cfg_utils.is_core_package(ls)
        elif lt == "datatype" and ls not in is_core_datatype:
          is_core_datatype[ls] = cfg_utils.is_core_label(ls, types_to_pkgs)
        elif lt == "function" and ls not in is_core_function:
          is_core_function[ls] = cfg_utils.is_core_label(ls, funcs_to_pkgs)

  for lt, ls in labels.items():
    ls = sorted(ls.items(), key=lambda e: e[1], reverse=True)
    labels[lt] = ls

  labels["core_package"] = is_core_package
  labels["core_datatype"] = is_core_datatype
  labels["core_function"] = is_core_function

  return labels

@utils.cached(
  DATA_DIR / "cfg_dims",
  lambda _, limit_id=default_limit_id, split_id=None, mode=default_mode: (
    f"dims_{mode}_{get_limit_name(limit_id)}"
    + ("" if split_id is None else f"_{split_id}")),
  "pretty_json")
def create_graph_dims(
  graphs, limit_id=default_limit_id, split_id=None, mode=default_mode):
  labels = collect_node_label_histogram(graphs, split_id, mode)
  node_dim_count = 0
  node_dims = dict()
  edge_dims = dict()
  if isinstance(limit_id, dict):
    dim_limits = limit_id
  else:
    dim_limits = get_dim_limit_dict()[limit_id]
  only_core_packages = dim_limits.get("only_core_packages", False)
  if only_core_packages:
    core_packages = labels["core_package"]
    core_datatypes = labels["core_datatype"]
    core_functions = labels["core_function"]

  for lt in cfg_utils.node_label_types.keys():
    ls = labels[lt]

    if only_core_packages:
      if lt == "package":
        ls = fy.filter(lambda l: core_packages.get(l[0], False), ls)
      elif lt == "datatype":
        ls = fy.filter(lambda l: core_datatypes.get(l[0], False), ls)
      elif lt == "function":
        ls = fy.filter(lambda l: core_functions.get(l[0], False), ls)

    d = dict()
    node_dims[lt] = d
    limit = dim_limits.get(lt, None)
    if limit is not False and lt not in no_default_label_subtype:
      d[""] = node_dim_count
      node_dim_count += 1

    if limit is not None:
      if limit is False:
        ls = []
      elif isinstance(limit, set):
        ls = fy.filter(lambda t: t[0] in limit, ls)
      elif limit is not True:
        ls = fy.take(limit, ls)

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

def merge_dims(dims_list):
  merged_node_dims = dict()
  remap_idxs = []
  new_idx = 0
  for dims in dims_list:
    remap_idx = np.empty(dims["node_label_count"], dtype=np.int32)
    for label_type, sublabels in dims["node_labels"].items():
      if label_type not in merged_node_dims:
        merged_sublabels = dict()
        merged_node_dims[label_type] = merged_sublabels
      else:
        merged_sublabels = merged_node_dims[label_type]
      for sublabel, idx in sublabels.items():
        if sublabel not in merged_sublabels:
          merged_sublabels[sublabel] = new_idx
          remap_idx[idx] = new_idx
          new_idx += 1
        else:
          remap_idx[idx] = merged_sublabels[sublabel]
    remap_idxs.append(remap_idx)

  dims0 = dims_list[0]
  return dict(
    node_labels=merged_node_dims,
    node_label_count=new_idx,
    edge_labels=dims0["edge_labels"],
    edge_label_count=dims0["edge_label_count"]
  ), remap_idxs

def apply_remap_idx(
  X, remap_idx, node_label_count, in_enc=None, with_marked_idx=False):
  offset = 0
  if with_marked_idx:
    offset += 1
  if in_enc == "wl2":
    offset += 3
  old_node_label_count = remap_idx.size
  old_final_offset = offset + old_node_label_count
  final_offset = offset + node_label_count
  node_label_diff = node_label_count - old_node_label_count
  X_remapped = np.zeros(
    X.shape[:-1] + (X.shape[-1] + node_label_diff,), dtype=X.dtype)
  X_remapped[..., :offset] = X[..., :offset]
  X_remapped[..., remap_idx + offset] = X[..., offset:old_final_offset]
  X_remapped[..., final_offset:] = X[..., old_final_offset:]
  return X_remapped

def apply_remap_idxs(Xs, remap_idxs, node_label_count, in_enc, with_marked_idx=True):
  return [
    apply_remap_idx(X, remap_idx, node_label_count, in_enc, with_marked_idx)
    for X, remap_idx in zip(Xs, remap_idxs)]

def get_node_label_dims(node_dims, node_data):
  labels = node_data["labels"]
  dims = []

  for label in labels:
    lt, ls = label
    d = node_dims[lt]
    if ls in d:
      dims.append(d[ls])
    elif "" in d:
      dims.append(d[""])
    elif lt == "type":
      return False

  return np.array(dims)

def get_edge_label_dim(edge_dims, edge_data):
  return edge_dims[edge_data["label"]]

def dims_to_labels(dims, in_enc, with_marked_idx=True, with_graph_type=True):
  res = [("marked", None)] if with_marked_idx else []
  node_dims_inv = [None] * dims["node_label_count"]

  for bt, bd in dims["node_labels"].items():
    for name, dim in bd.items():
      node_dims_inv[dim] = (bt, name)

  if in_enc == "wl2":
    edge_dims_inv = [None] * dims["edge_label_count"]
    for et, dim in dims["edge_labels"].items():
      edge_dims_inv[dim] = ("edge", et)

    res += [
      ("metatype", "node"),
      ("metatype", "edge"),
      ("metatype", "indirect")
    ]
    res += node_dims_inv
    res += edge_dims_inv
  else:
    res += node_dims_inv

  if with_graph_type:
    res += [("context", cfg_type) for cfg_type in cfg_utils.cfg_types]

  return res

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

@utils.cached(
  DATA_DIR / "wl2",
  lambda graphs, dims, radius=1, split_id=None: (
    f"wl2_r{radius}" + ("" if split_id is None else "_" + split_id)))
def wl2_encode_graphs(graphs, dims, radius=1, split_id=None):
  enc_graphs = np.empty(len(graphs), dtype="O")
  node_label_fn = fy.partial(get_node_label_dims, dims["node_labels"])
  edge_label_fn = fy.partial(get_edge_label_dim, dims["edge_labels"])
  node_label_count = dims["node_label_count"]
  edge_label_count = dims["edge_label_count"]

  for i, g in enumerate(graphs):
    enc_g = wl2.encode_graph(
      g, radius=radius,
      node_label_count=node_label_count,
      edge_label_count=edge_label_count,
      node_label_fn=node_label_fn,
      edge_label_fn=edge_label_fn,
      with_marked_node=True)
    enc_g["graph_X"] = graph_features[g.cfg_type]
    enc_graphs[i] = enc_g

  return enc_graphs

def wl1_tf_dataset(
  dataset, dims, multirefs=True, split_id=None,
  batch_size_limit=None, batch_space_limit=None):
  if isinstance(dataset, tuple):
    graphs, targets = dataset
  else:
    graphs = dataset
    targets = None
  encoded_graphs = wl1_encode_graphs(graphs, dims, multirefs, split_id)

  node_label_count = dims["node_label_count"]
  edge_label_count = dims["edge_label_count"]
  ds_batcher = wl1.WL1Batcher(
    batch_size_limit=batch_size_limit,
    batch_space_limit=batch_space_limit)
  if targets is not None:
    ds_batcher = trans.tuple(
      ds_batcher,
      trans.pair(batcher.Batcher.identity))
    gen = ds_batcher.batch_generator((encoded_graphs, targets))
    out_enc = "int32_pair"
  else:
    gen = ds_batcher.batch_generator(encoded_graphs)
    out_enc = None

  in_enc = "mwl1" if multirefs else "wl1"
  in_meta = dict(
    node_label_count=node_label_count,
    edge_label_count=edge_label_count,
    graph_feature_dim=len(graph_features),
    with_marked_node=True)
  out_meta = dict()
  return tf.make_dataset(gen, in_enc, in_meta, out_enc, out_meta)

def wl2_tf_dataset(
  dataset, dims, radius=1, split_id=None,
  batch_size_limit=None, batch_space_limit=None):
  if isinstance(dataset, tuple):
    graphs, targets = dataset
  else:
    graphs = dataset
    targets = None
  encoded_graphs = wl2_encode_graphs(graphs, dims, radius, split_id)

  node_label_count = dims["node_label_count"]
  edge_label_count = dims["edge_label_count"]
  ds_batcher = wl2.WL2Batcher(
    batch_size_limit=batch_size_limit,
    batch_space_limit=batch_space_limit)
  if targets is not None:
    ds_batcher = trans.tuple(
      ds_batcher,
      trans.pair(batcher.Batcher.identity))
    gen = ds_batcher.batch_generator((encoded_graphs, targets))
    out_enc = "int32_pair"
  else:
    gen = ds_batcher.batch_generator(encoded_graphs)
    out_enc = None

  in_enc = "wl2"
  in_meta = dict(
    node_label_count=node_label_count,
    edge_label_count=edge_label_count,
    graph_feature_dim=len(graph_features),
    with_marked_node=True)
  out_meta = dict()
  return tf.make_dataset(gen, in_enc, in_meta, out_enc, out_meta)

def node_set_tf_dataset(
  dataset, dims, split_id=None,
  batch_size_limit=None, batch_space_limit=None):
  if isinstance(dataset, tuple):
    graphs, targets = dataset
  else:
    graphs = dataset
    targets = None
  encoded_graphs = wl1_encode_graphs(graphs, dims, False, split_id)

  node_label_count = dims["node_label_count"]
  ds_batcher = wl1.SetBatcher(
    batch_size_limit=batch_size_limit,
    batch_space_limit=batch_space_limit)
  if targets is not None:
    ds_batcher = trans.tuple(
      ds_batcher,
      trans.pair(batcher.Batcher.identity))
    gen = ds_batcher.batch_generator((encoded_graphs, targets))
    out_enc = "int32_pair"
  else:
    gen = ds_batcher.batch_generator(encoded_graphs)
    out_enc = None

  in_meta = dict(
    node_label_count=node_label_count,
    graph_feature_dim=len(graph_features),
    with_marked_node=True)
  out_meta = dict()
  return tf.make_dataset(gen, "node_set", in_meta, out_enc, out_meta)

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
  wl2=wl2_tf_dataset,
  node_set=node_set_tf_dataset
)

def get_encoded_dataset_slices(
  dataset, enc, split_idxs,
  outer_i=0, inner_i=0,
  limit_id=default_limit_id, mode=default_mode, **kwargs):
  split_id = f"{outer_i}_{inner_i}"
  encoder = dataset_encoders[enc]
  train_slice, val_slice, test_slice = get_dataset_slices(
    dataset, split_idxs, outer_i, inner_i)
  train_dims = create_graph_dims(
    train_slice[0], limit_id, f"{split_id}_train", mode)

  train_ds = encoder(
    train_slice, train_dims,
    split_id=f"{mode}_{get_limit_name(limit_id)}_{split_id}_train", **kwargs)
  val_ds = encoder(
    val_slice, train_dims,
    split_id=f"{mode}_{get_limit_name(limit_id)}_{split_id}_val", **kwargs)
  test_ds = encoder(
    test_slice, train_dims,
    split_id=f"{mode}_{get_limit_name(limit_id)}_{split_id}_test", **kwargs)
  return train_dims, train_ds, val_ds, test_ds
