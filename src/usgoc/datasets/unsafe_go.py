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
    res.add(ct["type"])
    ctid = ct["elem"]
    ct = types[ctid]

  # Reduce tuple types to set of contained types:
  if ct["type"] == "Tuple":
    for field in ct["fields"]:
      res |= type_to_labels(types, field["type"])
  else:
    res.add("t=" + ct["name"])
  return res

def func_to_labels(funcs, pkgs, fid):
  if fid == -1:
    return set()

  func = funcs[fid]
  pid = func["package"]
  pname = pkgs[pid]["path"] + "." if pid >= 0 else ""
  fname = pname + func["name"]

  return {"f=" + fname}

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
      p = "b"
      succ = ["left", "right"]
      # simplify op pairs, since operands are not preserved:
      if op == ">":
        op = "<"
      elif op == ">=":
        op = "<="
    else:
      p = "u"
      succ = ["target"]
    ops.add(p + "=" + op)
    return ops, succ
  elif k == "statement" and t == "crement":
    ops.add("u" + "=" + s["operation"])
    return ops, ["target"]
  elif k == "expression" and t == "cast":
    ops |= t2l(s["coerced-to"].get("go-type", -1))
    return ops, ["target"]
  elif k == "expression" and t == "call":
    ops |= t2l(s.get("go-type", -1))
    func = s["function"]
    if func["kind"] == "expression" and func["type"] == "identifier":
      fid = func["value"].get("function", -1)
    elif func["kind"] == "expression" and func["type"] == "selector":
      fid = func["field"].get("function", -1)
    ops |= f2l(fid)
    return ops, ["arguments", "function"]

  return ops, True

def find_operations_in_ast(t2l, f2l, ast):
  return utils.walk_nested(
    fy.partial(_walk_ast_for_ops, t2l, f2l), set(), ast)

def ast_to_labels(t2l, f2l, ast):
  if ast is None:
    return set()

  res = find_operations_in_ast(t2l, f2l, ast)
  res.add(ast["type"])

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
    labels = {"block"}
    if block["entry"]:
      labels.add("entry")
    elif block["exit"]:
      labels.add("exit")
    labels |= ast_to_labels(t2l, f2l, block["ast"])
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
    elif btype in {"assign", "define"}:
      # Find nested assign targets:
      if btype == "assign" and len(assign_vars) == 0:
        left_vars = find_vars_in_selectors(ast["left"])
        assign_vars |= left_vars["var_roots"]
        up_vars |= left_vars["var_used"]
        var_contains |= left_vars["contains"]
        leftover_vars = find_vars_in_expr(left_vars["missing_exprs"])
        use_vars |= leftover_vars["vars"]
        var_contains |= leftover_vars["contains"]

      # Find variables on right side of assignment/definition:
      nested_vars = find_vars_in_expr(ast["right"])

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
      key = "flow" if i == 0 else "alt-flow"
      g.add_edge(b, block_ids[s], key=key, label=key)

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
