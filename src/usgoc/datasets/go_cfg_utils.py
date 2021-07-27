from collections import defaultdict
import funcy as fy
import networkx as nx

import usgoc.utils as utils

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

  # Reduce tuple and struct types to set of contained types:
  if ct["type"] in {"Tuple", "Struct"}:
    for field in ct["fields"]:
      res |= type_to_labels(
        types, vars, field["type"], selfrefs, pkg_registry,
        parents=parents, visited=visited)
  # Reduce maps to their key/value types:
  elif ct["type"] == "Map":
    res |= type_to_labels(
      types, vars, ct["key"], selfrefs, pkg_registry,
      parents=parents, visited=visited)
    res |= type_to_labels(
      types, vars, ct["elem"], selfrefs, pkg_registry,
      parents=parents, visited=visited)
  # Compute package dependencies of signatures and interfaces:
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
  elif k == "literal" and t == "composite":
    ops |= t2l(s.get("go-type", -1))
    return ops, ["values"]
  elif k == "decl" and t == "type-alias":
    for b_outer in s["binds"]:
      b = b_outer["value"]
      ops |= t2l(b.get("go-type", -1))
      if b["kind"] == "type" and b["type"] == "struct":
        for field in b["fields"]:
          ops |= t2l(field["declared-type"].get("go-type", -1))
    return ops, False
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
  elif k == "constant":
    ops |= t2l(s.get("go-type", -1))
    return ops, False

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

def cfg_to_graph(cfg, mark_line=None, module=None, mode=None):
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

  # Add block nodes:
  for i, block in enumerate(blocks):
    labels = {("type", "block")}
    if block["entry"]:
      if mark_block is None:
        mark_block = n
      labels.add(("blocktype", "entry"))
    elif block["exit"]:
      labels.add(("blocktype", "exit"))
    labels |= ast_to_labels(t2l, f2l, block["ast"])
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
    n += 1

  if mark_line is not None and mark_block is not None:
    b = g.nodes[mark_block]
    b["marked"] = True
    b["color"] = "red"

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
