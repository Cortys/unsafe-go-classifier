import glob
import json
from pathlib import Path
from collections import defaultdict
import funcy as fy
import networkx as nx
import numpy as np

import usgoc.utils as utils
import usgoc.preprocessing.graph.wl1 as wl1
import usgoc.preprocessing.transformer as trans
import usgoc.preprocessing.batcher as batcher
import usgoc.preprocessing.tf as tf
import usgoc.preprocessing.split as ps

RAW_DATASET_PATTERN = "/app/raw/unsafe-go-dataset/**/*.json"
DATA_DIR = Path("/app/data/unsafe-go-dataset")

node_label_types = dict(
  type="",
  blocktype="b",
  vartype="v",
  varname="n",
  datatype="t",
  datatype_flag="tf",
  function="f",
  builtin_function="fb",
  binary_op="ob",
  unary_op="ou",
)
no_ellipsis_types = {
  "type", "blocktype", "vartype", "builtin_function",
  "binary_op", "unary_op", "datatype_flag"
}
no_default_label_subtype = {
  "type", "blocktype", "vartype"
}
node_label_type_dim_count = dict(
  varname=15,
  datatype=63,
  function=63
)
edge_labels = [
  "flow",
  "alt-flow",
  "decl",
  "assign",
  "update",
  "use",
  "call",
  "contains"
]

def load_filenames():
  return glob.glob(RAW_DATASET_PATTERN, recursive=True)

def load_raw():
  files = load_filenames()
  res = []
  for f in files:
    with open(f, "r") as fp:
      res.append(json.load(fp))
  return res

def get_node_label(labels):
  def l2s(label):
    lt, ls = label
    if lt == "type":
      return ls

    if lt not in no_ellipsis_types and len(ls) > 12:
      ls = ls[:4] + "…" + ls[-7:]

    return f"{node_label_types[lt]}[{ls}]"

  labels = sorted(labels, key=lambda l: node_label_types[l[0]])
  return "\n".join(fy.map(l2s, labels))

def ast_type(ast):
  if ast is None:
    return None
  return ast["type"]

def type_to_labels(types, tid):
  if tid == -1:
    return set()
  ctid = tid
  ct = types[ctid]
  while ct["underlying"] != ctid:
    ctid = ct["underlying"]
    ct = types[ctid]

  res = set()

  # Flatten pointer/array/slice types and add corresponding flags instead:
  while ct["type"] in {"Pointer", "Slice", "Array"}:
    res.add(("datatype_flag", ct["type"]))
    ctid = ct["elem"]
    ct = types[ctid]

  # Reduce tuple types to set of contained types:
  if ct["type"] == "Tuple":
    for field in ct["fields"]:
      res |= type_to_labels(types, field["type"])
  else:
    res.add(("datatype", ct["name"]))
    res.add(("datatype_flag", ct["type"]))
  return res

def func_to_labels(funcs, pkgs, fid):
  if fid == -1:
    return set()

  if isinstance(fid, str):
    return {("builtin_function", fid)}

  func = funcs[fid]
  pid = func["package"]
  pname = pkgs[pid]["path"] + "." if pid >= 0 else ""
  fname = pname + func["name"]

  return {("function", fname)}

def _walk_var_selector_chain(state, s):
  if not isinstance(s, dict):
    return state, False

  at_root = state["at_root"]
  state["at_root"] = False
  k = s.get("kind", None)
  t = s.get("type", None)
  si = {"selector", "identifier"}

  if k == "expression" and t in si:
    if t == "selector":
      v = s["field"].get("variable", -1)
    else:
      v = s["value"].get("variable", -1)
    if v is not None and v >= 0:
      if at_root:
        state["var_root"] = v
      if state["edge_target"] is not None:
        state["contains"].add((v, state["edge_target"]))
      state["edge_target"] = v
      state["var_used"].add(v)
      return state, ["target"]
  elif k == "expression" and t == "index":
    state["missing_exprs"].append(s["index"])
    return state, ["target"]

  return state, False

def find_vars_in_selectors(exprs):
  var_roots = set()
  var_used = set()
  contains = set()
  missing_exprs = []

  for expr in exprs:
    state = utils.walk_nested(_walk_var_selector_chain, dict(
      at_root=True,
      var_root=None,
      edge_target=None,
      var_used=var_used,
      contains=contains,
      missing_exprs=missing_exprs), expr)
    var_used = state["var_used"]
    contains = state["contains"]
    missing_exprs = state["missing_exprs"]
    if state["var_root"] is not None:
      var_roots.add(state["var_root"])

  return dict(
    var_roots=var_roots,
    var_used=var_used,
    contains=contains,
    missing_exprs=missing_exprs)

def _walk_expr_for_vars(state, s):
  if not isinstance(s, dict):
    return state, True

  k = s.get("kind", None)
  t = s.get("type", None)
  si = {"selector", "identifier"}

  if k == "expression" and t in si:
    vars = find_vars_in_selectors([s])
    state["vars"] |= vars["var_used"]
    state["contains"] |= vars["contains"]
    return state, vars["missing_exprs"]
  elif k == "expression" and t == "call":
    func = s["function"]
    if func["kind"] == "expression" and func["type"] in si:
      if func["type"] == "selector":
        v = func["field"].get("variable", -1)
      else:
        v = func["value"].get("variable", -1)
      if v is not None and v >= 0:
        state["called_vars"].add(v)
        return state, ["function", "arguments"]

  return state, True

def find_vars_in_expr(expr):
  state = utils.walk_nested(
    _walk_expr_for_vars, dict(
      vars=set(),
      called_vars=set(),
      contains=set(),
    ), expr)

  return state

def _walk_ast_for_ops(t2l, f2l, ops, s):
  if not isinstance(s, dict):
    return ops, True

  k = s.get("kind", None)
  t = s.get("type", None)
  if k == "expression" and t in {"unary", "binary"}:
    op = s["operator"]
    if t == "binary":
      p = "binary_op"
      succ = ["left", "right"]
      # simplify op pairs, since operand order is not preserved anyway:
      if op == ">":
        op = "<"
      elif op == ">=":
        op = "<="
    else:
      p = "unary_op"
      succ = ["target"]
    ops.add((p, op))
    return ops, succ
  elif k == "statement" and t == "crement":
    ops.add(("unary_op", s["operation"]))
    return ops, ["target"]
  elif k == "statement" and t == "assign-operator":
    ops.add(("binary_op", s["operator"]))
    return ops, ["left", "right"]
  elif k == "expression" and t == "cast":
    ops |= t2l(s["coerced-to"].get("go-type", -1))
    return ops, ["target"]
  elif k == "expression" and t == "call":
    ops |= t2l(s.get("go-type", -1))
    func = s["function"]
    if func["kind"] == "expression" and func["type"] == "identifier":
      if func["value"]["ident-kind"] == "Builtin":
        fid = func["value"]["value"]
      else:
        fid = func["value"].get("function", -1)
    elif func["kind"] == "expression" and func["type"] == "selector":
      fid = func["field"].get("function", -1)
    ops |= f2l(fid)
    return ops, ["arguments", "function"]
  elif k == "expression" and t == "new":
    ops |= t2l(s.get("go-type", -1))
    ops |= f2l("new")
    return ops, ["argument"]

  return ops, True

def find_operations_in_ast(t2l, f2l, ast):
  return utils.walk_nested(
    fy.partial(_walk_ast_for_ops, t2l, f2l), set(), ast)

def ast_to_labels(t2l, f2l, ast):
  if ast is None:
    return set()

  res = find_operations_in_ast(t2l, f2l, ast)
  res.add(("blocktype", ast["type"]))

  return res

def cfg_to_graph(cfg):
  g = nx.MultiDiGraph()
  blocks = cfg["blocks"]
  vars = cfg["variables"]
  types = cfg["types"]
  pkgs = cfg["packages"]
  funcs = cfg["functions"]
  params = set(cfg["params"])
  receivers = set(cfg["receivers"])
  results = set(cfg["results"])
  n = 0
  block_ids = dict()
  var_ids = dict()
  t2l = utils.memoize(fy.partial(type_to_labels, types))
  f2l = utils.memoize(fy.partial(func_to_labels, funcs, pkgs))

  # Add block nodes:
  for i, block in enumerate(blocks):
    labels = {("type", "block")}
    if block["entry"]:
      labels.add(("blocktype", "entry"))
    elif block["exit"]:
      labels.add(("blocktype", "exit"))
    labels |= ast_to_labels(t2l, f2l, block["ast"])
    g.add_node(n, label=get_node_label(labels), labels=labels)
    block_ids[i] = n
    n += 1

  # Add variable nodes:
  for i, v in enumerate(vars):
    labels = {("type", "var")}
    if v["name"] != "":
      labels.add(("varname", v["name"]))
    if i in params:
      labels.add(("vartype", "param"))
    elif i in receivers:
      labels.add(("vartype", "receiver"))
    elif i in results:
      labels.add(("vartype", "result"))
    labels |= t2l(v["type"])
    g.add_node(n, label=get_node_label(labels), labels=labels)
    var_ids[i] = n
    n += 1

  # Add edges:
  for i, block in enumerate(blocks):
    b = block_ids[i]
    ast = block["ast"]
    btype = ast_type(ast)
    succs = block["successors"]
    assign_vars = set(block["assign-vars"])
    decl_vars = set(block["decl-vars"])
    up_vars = set(block["update-vars"])
    use_vars = set(block["use-vars"])
    call_vars = set()
    var_contains = set()
    nested_vars = None

    if block["entry"]:
      for param in params:
        g.add_edge(b, var_ids[param], key="decl", label="decl")
      for rec in receivers:
        g.add_edge(b, var_ids[rec], key="decl", label="decl")
      for res in results:
        g.add_edge(b, var_ids[res], key="decl", label="decl")

    if btype == "return":
      for res in results:
        if res not in use_vars:
          g.add_edge(b, var_ids[res], key="assign", label="assign")
          g.add_edge(b, var_ids[res], key="update", label="update")
    elif btype in {"assign", "assign-operator", "define"}:
      ls = ast["left"]
      rs = ast["right"]
      # Find nested assign targets (non-nested ones would be in assign_vars):
      if btype in {"assign", "assign-operator"} and len(assign_vars) < len(ls):
        left_vars = find_vars_in_selectors(ls)
        assign_vars |= left_vars["var_roots"]
        up_vars |= left_vars["var_used"]
        var_contains |= left_vars["contains"]
        leftover_vars = find_vars_in_expr(left_vars["missing_exprs"])
        use_vars |= leftover_vars["vars"]
        var_contains |= leftover_vars["contains"]

      # Find variables on right side of assignment/definition:
      nested_vars = find_vars_in_expr(rs)

    if nested_vars is None:
      # Find variable usages anywhere in AST:
      nested_vars = find_vars_in_expr(ast)
    use_vars |= nested_vars["vars"]
    call_vars |= nested_vars["called_vars"]
    var_contains |= nested_vars["contains"]

    for v1, v2 in var_contains:
      g.add_edge(
        var_ids[v1], var_ids[v2], key="contains", label="contains")
    for v in assign_vars:
      g.add_edge(b, var_ids[v], key="assign", label="assign")
    for v in decl_vars:
      g.add_edge(b, var_ids[v], key="decl", label="decl")
    for v in up_vars:
      g.add_edge(b, var_ids[v], key="update", label="update")
    for v in use_vars:
      g.add_edge(b, var_ids[v], key="use", label="use")
    for v in call_vars:
      g.add_edge(b, var_ids[v], key="use", label="call")
    for i, s in enumerate(succs):
      if btype == "switch":
        a = blocks[s]["ast"]
        is_default = a["type"] == "case-clause" and len(a["expressions"]) == 0
        key = "alt-flow" if is_default else "flow"
      else:
        key = "flow" if i == 0 else "alt-flow"
      g.add_edge(b, block_ids[s], key=key, label=key)

  # Prune unused/unreferenced variable nodes:
  for i in range(len(vars)):
    v = var_ids[i]
    if ("type", "var") in g.nodes[v]["labels"] and g.degree(v) == 0:
      g.remove_node(v)

  g.source_code = cfg["code"]

  return g

@utils.cached(DATA_DIR, "cfgs")
def raw_to_graphs(raw_dataset):
  graphs = np.empty(len(raw_dataset), dtype="O")
  for i, inst in enumerate(raw_dataset):
    graphs[i] = cfg_to_graph(inst["cfg"])

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
  labels = {type: defaultdict(lambda: 0) for type in node_label_types.keys()}

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
  lambda _, split_id=None: (
    f"dims{'' if split_id is None else '_' + split_id}"),
  "pretty_json")
def create_graph_dims(graphs, split_id=None):
  labels = collect_node_label_histogram(graphs, split_id)
  node_dim_count = 0
  node_dims = dict()
  edge_dims = dict()

  for lt in node_label_types.keys():
    ls = labels[lt]
    d = dict()
    node_dims[lt] = d
    if lt not in no_default_label_subtype:
      d[""] = node_dim_count
      node_dim_count += 1

    limit = node_label_type_dim_count.get(lt, None)
    if limit is not None:
      ls = ls[:limit]
    for lb, c in ls:
      d[lb] = node_dim_count
      node_dim_count += 1

  for i, ls in enumerate(edge_labels):
    edge_dims[ls] = i

  return dict(
    node_labels=node_dims,
    node_label_count=node_dim_count,
    edge_labels=edge_dims,
    edge_label_count=len(edge_labels))

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
    enc_graphs[i] = wl1.encode_graph(
      g,
      node_label_count=node_label_count,
      edge_label_count=edge_label_count,
      node_label_fn=node_label_fn,
      edge_label_fn=edge_label_fn,
      multirefs=multirefs)

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

  in_meta = dict(
    node_label_count=node_label_count,
    edge_label_count=edge_label_count,
    multirefs=multirefs)
  out_meta = dict()
  return tf.make_dataset(gen, "wl1", in_meta, "int32_pair", out_meta)

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

def wl1_tf_datasets(
  dataset, split_idxs, outer_i=0, inner_i=0, **kwargs):
  split_id = f"{outer_i}_{inner_i}"
  train_slice, val_slice, test_slice = get_dataset_slices(
    dataset, split_idxs, outer_i, inner_i)
  train_dims = create_graph_dims(train_slice[0], f"{split_id}_train")

  train_ds = wl1_tf_dataset(
    train_slice, train_dims, split_id=f"{split_id}_train", **kwargs)
  val_ds = wl1_tf_dataset(
    val_slice, train_dims, split_id=f"{split_id}_val", **kwargs)
  test_ds = wl1_tf_dataset(
    test_slice, train_dims, split_id=f"{split_id}_test", **kwargs)
  return train_dims, train_ds, val_ds, test_ds
