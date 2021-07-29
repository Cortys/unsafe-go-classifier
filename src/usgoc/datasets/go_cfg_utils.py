from collections import defaultdict
import funcy as fy
import networkx as nx

import usgoc.utils as utils

convert_modes = {"atomic_blocks", "split_blocks"}
cfg_types = ["function", "external", "type", "variable"]
node_label_types = dict(
  type="",
  blocktype="b",
  selfref="s",
  vartype="v",
  varname="n",
  datatype="t",
  datatype_flag="tf",
  function="f",
  builtin_function="fb",
  binary_op="ob",
  unary_op="ou",
  package="p"
)
no_ellipsis_types = {
  "type", "blocktype", "selfref",
  "vartype", "builtin_function",
  "binary_op", "unary_op", "datatype_flag"
}
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
semantic_names = {
  "_", "in", "out", "err", "ptr", "at", "to", "ok", "path",
  "fd", "key", "val", "newVal", "data", "typ", "size", "i"}

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

def is_core_package(pkg_path):
  if "." in pkg_path and not pkg_path.startswith("golang.org"):
    return False
  return True

def is_core_label(label, label_to_pkgs):
  pkgs = label_to_pkgs.get(label, None)
  return pkgs is None or fy.all(is_core_package, pkgs)

def is_semantic_name(name):
  if len(name) == 1:
    return True
  elif name in semantic_names:
    return True
  elif name[0] == "[":
    return True
  else:
    return False

def type_to_labels(
  types, vars, tid,
  selfrefs=set(), pkg_registry=None,
  parents=None, visited=None):

  if tid == -1:
    return set()
  if parents is None:
    parents = set()
  if visited is None:
    visited = set()

  # Break recursive type loops and
  # cut type branches that are isomorphic to a visited branch:
  if tid in parents or tid in visited:
    # Propagate referenced packages to parents:
    pkgs = pkg_registry.get(types[tid]["name"])
    if pkgs is not None:
      for p in parents:
        pkg_registry[types[p]["name"]].update(pkgs)
    return set()

  tids = {tid}
  ctid = tid
  ct = types[ctid]
  res = set()

  while True:
    if ctid in selfrefs:
      res.add(("selfref", "type"))

    if ct["type"] == "Named":
      res.add(("datatype", ct["name"]))
      if pkg_registry is not None:
        ct_pkg = ct["package"]
        for p in tids | parents:
          pkg_registry[types[p]["name"]].add(ct_pkg)
      res.add(("datatype_flag", ct["type"]))

    if ct["underlying"] != ctid:
      ctid = ct["underlying"]
      tids.add(ctid)
      ct = types[ctid]
    # Flatten pointer/array/slice types and add corresponding flags instead:
    elif ct["type"] in {"Pointer", "Slice", "Array", "Chan"}:
      res.add(("datatype_flag", ct["type"]))
      if ct["type"] == "Chan":
        res.add(("datatype_flag", f"Chan_{ct['dir']}"))
      ctid = ct["elem"]
      tids.add(ctid)
      ct = types[ctid]
    else:
      break

  res.add(("datatype", ct["name"]))
  res.add(("datatype_flag", ct["type"]))
  parents = parents | tids
  visited.update(tids)

  # Compute package dependencies of nested types:
  if ct["type"] in {"Tuple", "Struct"}:
    for field in ct["fields"]:
      type_to_labels(
        types, vars, field["type"], selfrefs, pkg_registry,
        parents=parents, visited=visited)
  # Reduce maps to their key/value types:
  elif ct["type"] == "Map":
    type_to_labels(
      types, vars, ct["key"], selfrefs, pkg_registry,
      parents=parents, visited=visited)
    type_to_labels(
      types, vars, ct["elem"], selfrefs, pkg_registry,
      parents=parents, visited=visited)
  elif ct["type"] == "Interface":
    for method in ct["methods"]:
      type_to_labels(
        types, vars, method["type"], selfrefs, pkg_registry,
        parents=parents, visited=visited)
  elif ct["type"] == "Signature":
    recv_vid = ct["recv"]
    if recv_vid != -1:
      recv_tid = vars[recv_vid]["type"]
      type_to_labels(
        types, vars, recv_tid, selfrefs, pkg_registry,
        parents=parents, visited=visited)
    type_to_labels(
      types, vars, ct["params"], selfrefs, pkg_registry,
      parents=parents, visited=visited)
    type_to_labels(
      types, vars, ct["results"], selfrefs, pkg_registry,
      parents=parents, visited=visited)

  return res

def pkg_to_labels(pkgs, pid, cfg_pkg=-1, cfg_module=None):
  if pid == -1:
    return set()
  pkg_path = pkgs[pid]["path"]
  res = {("package", pkg_path)}

  if pid == cfg_pkg:
    res.add(("selfref", "package"))
    res.add(("selfref", "module"))
  elif cfg_module is not None and pkg_path.startswith(cfg_module):
    res.add(("selfref", "module"))

  return res

def func_to_labels(
  funcs, pkgs, fid,
  selfrefs=set(), cfg_pkg=-1, cfg_module=None,
  pkg_registry=None, with_pkg=True):
  if fid == -1:
    return set()

  if isinstance(fid, str):
    return {("builtin_function", fid)}

  func = funcs[fid]
  pid = func["package"]
  res = set()
  if pid >= 0:
    pkg_path = pkgs[pid]["path"]
    fname = pkg_path + "." + func["name"]
    if with_pkg:
      res |= pkg_to_labels(
        pkgs, pid, cfg_pkg, cfg_module)
    if pkg_registry is not None:
      pkg_registry[fname].add(pid)
  else:
    fname = func["name"]

  res.add(("function", fname))

  if fid in selfrefs:
    res.add(("selfref", "function"))

  return res

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
        state["root_var"] = v
      if state["edge_target"] is not None:
        state["contains"].add((v, state["edge_target"]))
      state["edge_target"] = v
      state["used_vars"].add(v)
      return state, ["target"]
  elif k == "expression" and t == "index":
    state["missing_exprs"].append(s["index"])
    return state, ["target"]

  return state, False

def find_vars_in_selectors(exprs):
  root_vars = set()
  used_vars = set()
  contains = set()
  missing_exprs = []

  for expr in exprs:
    state = utils.walk_nested(_walk_var_selector_chain, dict(
      at_root=True,
      root_var=None,
      edge_target=None,
      used_vars=used_vars,
      contains=contains,
      missing_exprs=missing_exprs), expr)
    used_vars = state["used_vars"]
    contains = state["contains"]
    missing_exprs = state["missing_exprs"]
    if state["root_var"] is not None:
      root_vars.add(state["root_var"])

  return dict(
    root_vars=root_vars,
    used_vars=used_vars,
    contains=contains,
    missing_exprs=missing_exprs)

def _walk_ast(state, s, t2l, f2l, split_blocks=False):
  if not isinstance(s, dict):
    return state, True

  si = {"selector", "identifier"}
  at_root = state["at_root"]
  state["at_root"] = False
  labels: dict = state["labels"]
  k = s.get("kind", None)
  t = s.get("type", None)

  if split_blocks and not at_root:
    if (k == "expression" and t in {"unary", "binary"}) \
        or (k == "literal" and t == "composite") \
        or (k == "expression" and t == "cast") \
        or (k == "expression" and t == "call") \
        or (k == "expression" and t == "new"):
      state["subblock_asts"].append(s)
      return state, False

  # Statement and declaration handlers (i.e. block level asts):
  if k == "statement" and t in {"assign", "assign-operator", "define"}:
    succs = [s["right"]]
    if t in {"assign", "assign-operator"}:  # Var defs. are assumed to be known
      if t == "assign-operator":
        labels.add(("binary_op", s["operator"]))

      left_vars = find_vars_in_selectors(s["left"])
      state["assigned_vars"] |= left_vars["root_vars"]
      state["updated_vars"] |= left_vars["used_vars"]
      state["var_contains"] |= left_vars["contains"]
      succs += left_vars["missing_exprs"]

    return state, succs
  elif k == "statement" and t == "crement":
    labels.add(("unary_op", s["operation"]))
    return state, ["target"]
  elif k == "decl" and t == "type-alias":
    for b_outer in s["binds"]:
      b = b_outer["value"]
      labels.update(t2l(b.get("go-type", -1)))
      if b["kind"] == "type" and b["type"] == "struct":
        for field in b["fields"]:
          labels.update(t2l(field["declared-type"].get("go-type", -1)))
    return state, False
  # Expression handlers (i.e. typically nested asts):
  elif k == "expression" and t in si:
    vars = find_vars_in_selectors([s])
    state["used_vars"] |= vars["used_vars"]
    state["var_contains"] |= vars["contains"]
    return state, vars["missing_exprs"]
  elif k == "expression" and t in {"unary", "binary"}:
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
    labels.add((p, op))
    return state, succ
  elif k == "literal" and t == "composite":
    labels.update(t2l(s.get("go-type", -1)))
    return state, ["values"]
  elif k == "expression" and t == "cast":
    labels.update(t2l(s["coerced-to"].get("go-type", -1)))
    return state, ["target"]
  elif k == "expression" and t == "call":
    labels.update(t2l(s.get("go-type", -1)))
    func = s["function"]
    fid = -1
    fvid = -1
    if func["kind"] == "expression" and func["type"] == "identifier":
      if func["value"]["ident-kind"] == "Builtin":
        fid = func["value"]["value"]
      else:
        fid = func["value"].get("function", -1)
        fvid = func["value"].get("variable", -1)
    elif func["kind"] == "expression" and func["type"] == "selector":
      fid = func["field"].get("function", -1)
      fvid = func["field"].get("variable", -1)
    if fid != -1:
      labels.update(f2l(fid))
    elif fvid != -1:
      state["called_vars"].add(fvid)
    return state, ["arguments", "function"]
  elif k == "expression" and t == "new":
    labels.update(t2l(s.get("go-type", -1)))
    labels.update(f2l("new"))
    return state, ["argument"]
  elif k == "constant":
    labels.update(t2l(s.get("go-type", -1)))
    return state, False

  return state, True

def walk_ast(t2l, f2l, ast, split_blocks=False):
  if ast is None:
    labels = set()
  else:
    labels = {("blocktype", ast["type"])}

  return utils.walk_nested(
    _walk_ast, dict(
      at_root=True,
      labels=labels,
      assigned_vars=set(),
      updated_vars=set(),
      used_vars=set(),
      called_vars=set(),
      var_contains=set(),
      subblock_asts=[]
    ), ast, t2l=t2l, f2l=f2l, split_blocks=split_blocks)

def cfg_to_graph(cfg, mark_line=None, module=None, mode=None):
  assert mode in convert_modes, f"Unknown CFG conversion mode '{mode}'."
  split_blocks = mode == "split_blocks"
  g = nx.MultiDiGraph()
  cfg_type = cfg["type"]
  cfg_package = cfg["package"]
  defined = set(cfg["defines"])
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
  type_pkg_registry = defaultdict(set)
  func_pkg_registry = defaultdict(set)
  func_defined = set()
  var_defined = set()
  type_defined = set()

  if cfg_type in {"function", "external"}:
    func_defined = defined
  elif cfg_type == "variable":
    var_defined = defined
  elif cfg_type == "type":
    type_defined = defined

  t2l = utils.memoize(fy.partial(
    type_to_labels, types, vars, selfrefs=type_defined,
    pkg_registry=type_pkg_registry))
  f2l = utils.memoize(fy.partial(
    func_to_labels, funcs, pkgs,
    selfrefs=func_defined, cfg_pkg=cfg_package, cfg_module=module,
    pkg_registry=func_pkg_registry))
  p2l = utils.memoize(fy.partial(
    pkg_to_labels, pkgs, cfg_pkg=cfg_package, cfg_module=module))
  mark_block = None
  mark_block_lines = None

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

    if i in var_defined:
      labels.add(("selfref", "variable"))

    labels |= p2l(v["package"])
    labels |= t2l(v["type"])
    g.add_node(n, label=get_node_label(labels), labels=labels, marked=False)
    var_ids[i] = n
    n += 1

  # Add block nodes:
  for i, block in enumerate(blocks):
    labels = {("type", "block")}
    if block["entry"]:
      if mark_block is None:
        mark_block = n
      labels.add(("blocktype", "entry"))
    elif block["exit"]:
      labels.add(("blocktype", "exit"))
    walk_result = walk_ast(
      t2l, f2l, block["ast"], split_blocks=split_blocks)
    labels.update(walk_result["labels"])

    g.add_node(n, label=get_node_label(labels), labels=labels, marked=False)
    block_ids[i] = n

    if mark_line is not None:
      line_start = block["line-start"]
      line_end = block["line-end"]
      block_lines = line_end - line_start
      if line_start <= mark_line <= line_end \
          and (mark_block_lines is None or block_lines < mark_block_lines):
        mark_block = n
        mark_block_lines = block_lines

    b = block_ids[i]
    ast = block["ast"]
    btype = ast_type(ast)
    decl_vars = set(block["decl-vars"])
    assign_vars = walk_result["assigned_vars"]
    assign_vars.update(block["assign-vars"])
    up_vars = walk_result["updated_vars"]
    up_vars.update(block["update-vars"])
    use_vars = walk_result["used_vars"]
    call_vars = walk_result["called_vars"]
    var_contains = walk_result["var_contains"]
    if not split_blocks:
      use_vars.update(block["use-vars"])

    if block["entry"]:
      for param in params:
        g.add_edge(b, var_ids[param], key="decl", label="decl")
      for rec in receivers:
        g.add_edge(b, var_ids[rec], key="decl", label="decl")
      for res in results:
        g.add_edge(b, var_ids[res], key="decl", label="decl")
    elif btype == "return":
      for res in results:
        if res not in use_vars:
          g.add_edge(b, var_ids[res], key="assign", label="assign")
          g.add_edge(b, var_ids[res], key="update", label="update")

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
      g.add_edge(b, var_ids[v], key="call", label="call")

    n += 1

  if mark_line is not None and mark_block is not None:
    b = g.nodes[mark_block]
    b["marked"] = True
    b["color"] = "red"

  # Add flow edges:
  for i, block in enumerate(blocks):
    b = block_ids[i]
    btype = ast_type(ast)
    succs = block["successors"]

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

  types_to_pkgs = dict()
  funcs_to_pkgs = dict()

  for type, pids in type_pkg_registry.items():
    types_to_pkgs[type] = set(
      pkgs[pid]["path"] for pid in pids if pid != -1)

  for func, pids in func_pkg_registry.items():
    funcs_to_pkgs[func] = set(
      pkgs[pid]["path"] for pid in pids if pid != -1)

  g.source_code = cfg["code"]
  g.cfg_type = cfg_type
  g.package = pkgs[cfg_package]["path"]
  g.types_to_pkgs = types_to_pkgs
  g.funcs_to_pkgs = funcs_to_pkgs

  return g
