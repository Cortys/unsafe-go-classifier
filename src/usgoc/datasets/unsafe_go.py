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
  print(ctid)
  ct = types[ctid]
  while ct["underlying"] != ctid:
    ctid = ct["underlying"]
    ct = types[ctid]

  res = set()

  while ct["type"] in {"Pointer", "Slice", "Array"}:
    res.add(ct["type"])
    ctid = ct["elem"]
    ct = types[ctid]

  res.add(ct["name"])
  return res

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
  print("conv", cfg)
  t2l = utils.memoize(fy.partial(type_to_labels, types))

  for i, block in enumerate(blocks):
    labels = {"block"}
    if block["entry"]:
      labels.add("entry")
    elif block["exit"]:
      labels.add("exit")
    g.add_node(n, label=get_label(labels), labels=labels)
    block_ids[i] = n
    n += 1

  for i, v in enumerate(vars):
    labels = {"var"}
    if i in params:
      labels.add("param")
    elif i in receivers:
      labels.add("receiver")
    elif i in results:
      labels.add("results")
    labels |= t2l(v["type"])
    g.add_node(n, label=get_label(labels), labels=labels)
    var_ids[i] = n
    n += 1

  for i, block in enumerate(blocks):
    b = block_ids[i]
    succs = block["successors"]
    assign_vars = block["assign-vars"]
    decl_vars = block["decl-vars"]
    up_vars = block["update-vars"]
    use_vars = block["use-vars"]
    for v in assign_vars:
      g.add_edge(b, var_ids[v], key="assign", label="assign")
    for v in decl_vars:
      g.add_edge(b, var_ids[v], key="declare", label="declare")
    for v in up_vars:
      g.add_edge(b, var_ids[v], key="update", label="update")
    for v in use_vars:
      g.add_edge(b, var_ids[v], key="use", label="use")
    for i, s in enumerate(succs):
      key = "flow" if i == 0 else "alt-flow"
      g.add_edge(b, block_ids[s], key=key, label=key)

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
