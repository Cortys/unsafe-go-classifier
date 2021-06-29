import glob
import json
import funcy as fy
import networkx as nx
from pathlib import Path

import usgoc.utils as utils

RAW_DATASET_PATTERN = "/app/raw/unsafe-go-dataset/**/*.json"
DATA_DIR = Path("/app/data/unsafe-go-dataset")

def load_filenames():
  return glob.glob(RAW_DATASET_PATTERN, recursive=True)

def load_raw():
  files = load_filenames()
  res = []
  for f in files:
    with open(f, "r") as fp:
      res.append(json.load(fp))
  return res

@utils.cached(DATA_DIR, "labels", "pretty_json")
def collect_labels(raw_dataset):
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

def usage_to_target(usage, labels):
  label1s, label2s = labels
  return label1s[usage["label1"]], label2s[usage["label2"]]

def get_label(labels):
  return "\n".join(labels)

def type_to_labels(types, tid):
  if tid == -1:
    return set()
  ctid = tid
  ct = types[ctid]
  while ct["underlying"] != ctid:
    ctid = ct["underlying"]
    ct = types[ctid]

  res = set()

  while ct["type"] in {"Pointer", "Slice", "Array"}:
    res.add(ct["type"])
    ctid = ct["elem"]
    ct = types[ctid]

  res.add("t=" + ct["name"])
  return res

def ast_type(ast):
  if ast is None:
    return None
  return ast["type"]

def ast_to_labels(ast):
  if ast is None:
    return set()

  res = {ast["type"]}

  return res

def _find_var_target(state, s):
  if not isinstance(s, dict):
    return state, False

  at_root = state["at_root"]
  state["at_root"] = False

  if s["kind"] == "expression" and s["type"] == "selector":
    v = s["field"].get("variable", -1)
    if v is not None and v >= 0:
      if at_root:
        state["var_target"] = v
      if state["edge_target"] is not None:
        state["contains"].add((v, state["edge_target"]))
      state["edge_target"] = v
      state["var_updated"].add(v)
      return state, ["target"]
  elif s["kind"] == "expression" and s["type"] == "index":
    return state, ["target"]
  elif s["kind"] == "expression" and s["type"] == "identifier":
    v = s["value"].get("variable", -1)
    if v is not None and v >= 0:
      if at_root:
        state["var_target"] = v
      if state["edge_target"] is not None:
        state["contains"].add((v, state["edge_target"]))
      state["edge_target"] = v
      state["var_updated"].add(v)
      return state, ["target"]

  return state, False

def find_var_targets(exprs):
  var_targets = set()
  var_updated = set()
  contains = set()

  for expr in exprs:
    state = utils.walk_nested(_find_var_target, dict(
      at_root=True,
      var_target=None,
      edge_target=None,
      var_updated=var_updated,
      contains=contains), expr)
    var_updated = state["var_updated"]
    contains = state["contains"]
    if state["var_target"] is not None:
      var_targets.add(state["var_target"])

  return var_targets, var_updated, contains

def cfg_to_graph(cfg):
  g = nx.MultiDiGraph()
  blocks = cfg["blocks"]
  vars = cfg["variables"]
  types = cfg["types"]
  params = set(cfg["params"])
  receivers = set(cfg["receivers"])
  results = set(cfg["results"])
  n = 0
  block_ids = dict()
  var_ids = dict()
  t2l = utils.memoize(fy.partial(type_to_labels, types))

  # Add block nodes:
  for i, block in enumerate(blocks):
    labels = {"block"}
    if block["entry"]:
      labels.add("entry")
    elif block["exit"]:
      labels.add("exit")
    labels |= ast_to_labels(block["ast"])
    g.add_node(n, label=get_label(labels), labels=labels)
    block_ids[i] = n
    n += 1

  # Add variable nodes:
  for i, v in enumerate(vars):
    labels = {"var"}
    if i in params:
      labels.add("param")
    elif i in receivers:
      labels.add("receiver")
    elif i in results:
      labels.add("result")
    labels |= t2l(v["type"])
    g.add_node(n, label=get_label(labels), labels=labels)
    var_ids[i] = n
    n += 1

  # Add edges:
  for i, block in enumerate(blocks):
    b = block_ids[i]
    ast = block["ast"]
    btype = ast_type(ast)
    succs = block["successors"]
    assign_vars = block["assign-vars"]
    decl_vars = block["decl-vars"]
    up_vars = block["update-vars"]
    use_vars = block["use-vars"]
    for v in assign_vars:
      g.add_edge(b, var_ids[v], key="assign", label="assign")
    for v in decl_vars:
      g.add_edge(b, var_ids[v], key="decl", label="decl")
    for v in up_vars:
      g.add_edge(b, var_ids[v], key="update", label="update")
    for v in use_vars:
      g.add_edge(b, var_ids[v], key="use", label="use")
    for i, s in enumerate(succs):
      key = "flow" if i == 0 else "alt-flow"
      g.add_edge(b, block_ids[s], key=key, label=key)

    if block["entry"]:
      for param in params:
        g.add_edge(b, var_ids[param], key="decl", label="decl")
      for rec in receivers:
        g.add_edge(b, var_ids[rec], key="decl", label="decl")
      for res in results:
        g.add_edge(b, var_ids[res], key="decl", label="decl")

    if btype == "return":
      for res in results:
        g.add_edge(b, var_ids[res], key="assign", label="assign")
        g.add_edge(b, var_ids[res], key="update", label="update")
    else:
      if btype in {"assign", "define"}:
        # Find nested assign targets:
        if btype == "assign" and len(assign_vars) == 0:
          var_targets, var_updated, contains = find_var_targets(
            ast["left"])
          for v in var_targets:
            g.add_edge(b, var_ids[v], key="assign", label="assign")
          for v in var_updated:
            g.add_edge(b, var_ids[v], key="update", label="update")
          for (v1, v2) in contains:
            g.add_edge(
              var_ids[v1], var_ids[v2], key="contains", label="contains")

  # Prune unused/unreferenced variable nodes:
  for i in range(len(vars)):
    v = var_ids[i]
    if "var" in g.nodes[v]["labels"] and g.degree(v) == 0:
      g.remove_node(v)

  g.source_code = cfg["code"]

  return g

@utils.cached(DATA_DIR, "in_nx")
def raw_to_graphs(raw_dataset):
  return [cfg_to_graph(inst["cfg"]) for inst in raw_dataset]

@utils.cached(DATA_DIR, "out_ids")
def raw_to_usages(raw_dataset):
  labels = collect_labels(raw_dataset)
  return [usage_to_target(inst["usage"], labels) for inst in raw_dataset]

def load_dataset():
  raw_dataset = load_raw()
  return raw_to_graphs(raw_dataset), raw_to_usages(raw_dataset)
