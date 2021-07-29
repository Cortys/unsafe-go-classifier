import os
import json
import pickle
import inspect
import numbers
import contextlib
from collections.abc import Iterable
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import funcy as fy

PROJECT_ROOT = os.getenv("PROJECT_ROOT", ".")

def tolerant(f=None, only_named=True, ignore_varkwargs=False):
  if f is None:
    return lambda f: tolerant(f, only_named, ignore_varkwargs)

  if hasattr(f, "__tolerant__"):
    return f

  spec = inspect.getfullargspec(f.__init__ if inspect.isclass(f) else f)
  f_varargs = spec.varargs is not None
  f_varkws = not ignore_varkwargs and spec.varkw is not None

  if (only_named or f_varargs) and f_varkws:
    return f

  f_args = spec.args
  f_kwonlyargs = spec.kwonlyargs

  @fy.wraps(f)
  def wrapper(*args, **kwargs):
    if not (only_named or f_varargs):
      args = args[:len(f_args)]
    if not f_varkws:
      kwargs = fy.project(kwargs, f_args[len(args):] + f_kwonlyargs)

    return f(*args, **kwargs)

  wrapper.__tolerant__ = True

  return wrapper

def unwrap_method(f):
  return getattr(f, "__func__", f)

def tolerant_method(f):
  return tolerant(unwrap_method(f))

def select_prefixed_keys(map, prefix, include_others=False, target=None):
  if target is None:
    target = dict()

  for k, v in map.items():
    if k.startswith(prefix):
      target[k[len(prefix):]] = v
    elif include_others:
      target[k] = v

  return target

class NumpyEncoder(json.JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    if (
      isinstance(obj, np.float32)
      or isinstance(obj, np.float64)
      or isinstance(obj, np.int32)
      or isinstance(obj, np.int64)):
      return np.asscalar(obj)
    return super().default(obj)

class NumpyDecoder(json.JSONDecoder):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, object_hook=self.object_hook, **kwargs)

  def object_hook(self, obj):
    if isinstance(obj, list):
      if fy.all(lambda o: isinstance(o, numbers.Number), obj):
        return np.array(obj)
      else:
        return [self.object_hook(o) for o in obj]
    elif isinstance(obj, dict):
      for key in obj.keys():
        obj[key] = self.object_hook(obj[key])

    return obj

def obj_array(objects):
  a = np.empty(len(objects), dtype="O")
  a[:] = objects

  return a

@contextlib.contextmanager
def local_seed(seed):
  state = np.random.get_state()
  np.random.seed(seed)
  try:
    yield
  finally:
    np.random.set_state(state)

def vec_to_unit(feat):
  if len(feat) > 0 and isinstance(feat[0], str):
    feat = [len(f) / 10 for f in feat]
  u = 0
  for i, s in enumerate(np.clip(feat, 0, 1), 1):
    u += (2 ** -i) * s

  return u

def draw_graph(
  g, y=None, with_features=False, with_colors=True,
  edge_colors=False, label_colors=True, layout="spring"):
  if layout in {"dot", "neato"}:
    A = nx.nx_agraph.to_agraph(g)
    A.write(f"{PROJECT_ROOT}/plots/graph.dot")
    A.layout(layout)
    A.draw(f"{PROJECT_ROOT}/plots/graph.png")
    return

  plt.figure()

  if y is not None:
    plt.title('Label: {}'.format(y))

  cmap = plt.get_cmap("hsv")
  node_color = [
    vec_to_unit([d.get("label", 0)] if label_colors else d.get("features", []))
    for n, d in g.nodes(data=True)] if with_colors else "#1f78b4"
  edge_color = [
    vec_to_unit([d.get("label", 0)] if label_colors else d.get("features", []))
    for a, b, d in g.edges(data=True)] if edge_colors else "#000000"

  f = dict(
    spring=nx.draw_spring,
    planar=nx.draw_planar,
    spectral=nx.draw_spectral,
    circular=nx.draw_circular,
    random=nx.draw_random,
    kawai=nx.draw_kamada_kawai)[layout]

  if with_features:
    labels = {
      n: f"{n}:" + str(data.get("features"))
      for n, data in g.nodes(data=True)
    }
    f(
      g, labels=labels,
      node_color=node_color,
      edge_color=edge_color,
      vmin=0, vmax=1, cmap=cmap)
  else:
    f(
      g, with_labels=True,
      node_color=node_color,
      edge_color=edge_color,
      vmin=0, vmax=1, cmap=cmap)

  plt.show()

def draw_confusion_matrix(m, labels):
  fig, ax = plt.subplots()
  ax.imshow(m)
  ax.set_xticks(np.arange(len(labels)))
  ax.set_yticks(np.arange(len(labels)))
  ax.set_xticklabels(labels)
  ax.set_yticklabels(labels)
  plt.setp(
    ax.get_xticklabels(), rotation=45, ha="right",
    rotation_mode="anchor")

  for i in range(len(labels)):
    for j in range(len(labels)):
      ax.text(j, i, m[i, j], ha="center", va="center", color="w")
  fig.tight_layout()
  plt.show()

def graph_feature_dims(g):
  dim_node_features = 0
  dim_edge_features = 0

  for _, data in g.nodes(data=True):
    f = data.get("features")
    if f is not None:
      dim_node_features = len(f)
    break

  for _, _, data in g.edges(data=True):
    f = data.get("features")
    if f is not None:
      dim_edge_features = len(f)
    break

  return dim_node_features, dim_edge_features

def graphs_meta(graphs, labels=True):
  assert len(graphs) > 0
  if labels:
    n_labels = set()
    e_labels = set()
    max_node_count = 0
    max_edge_count = 0

    for g in graphs:
      if g.order() > max_node_count:
        max_node_count = g.order()
      if g.size() > max_edge_count:
        max_edge_count = g.size()
      for n, d in g.nodes(data=True):
        if "label" in d:
          n_labels.add(d["label"])

      for u, v, d in g.edges(data=True):
        if "label" in d:
          e_labels.add(d["label"])

    n_nl = (
      max(n_labels) if n_labels != set() else 0)
    if n_nl != 0 and min(n_labels) == 0:
      n_nl += 1

    n_el = (
      max(e_labels) if e_labels != set() else 0)
    if n_el != 0 and min(e_labels) == 0:
      n_el += 1
  else:
    n_nl = 0
    n_el = 0

  d_nf, d_ef = graph_feature_dims(graphs[0])

  return dict(
    node_feature_dim=d_nf,
    edge_feature_dim=d_ef,
    node_label_count=n_nl,
    edge_label_count=n_el,
    max_node_count=max_node_count,
    max_edge_count=max_edge_count
  )

def make_dir(dir):
  if not dir.exists():
    os.makedirs(dir)

  return dir


cache_format_types = {"binary", "text", "custom"}

def cache_format(loader, dumper, type="binary"):
  assert type in cache_format_types

  class CacheFormat:
    type = type
    load = staticmethod(loader)
    dump = staticmethod(dumper)

  return CacheFormat


cache_formats = dict(
  pickle=pickle,
  json=cache_format(
    fy.partial(json.load, cls=NumpyDecoder),
    fy.partial(json.dump, cls=NumpyEncoder),
    type="text"),
  pretty_json=cache_format(
    fy.partial(json.load, cls=NumpyDecoder),
    fy.partial(json.dump, indent="\t", cls=NumpyEncoder),
    type="text")
)

def register_cache_format(format, load, dump, type="binary"):
  cache_formats[format] = cache_format(load, dump, type)
  return format

def cache_read(file, format="pickle"):
  assert format in cache_formats, f"Unknown format '{format}'."
  cache_format = cache_formats[format]
  type = "binary" if format == "pickle" else cache_format.type

  if type == "custom":
    return cache_format.load(file)
  else:
    with open(file, "rb" if type == "binary" else "r") as f:
      return cache_format.load(f)

def cache_write(file, data, format="pickle"):
  assert format in cache_formats, f"Unknown format '{format}'."
  cache_format = cache_formats[format]
  type = "binary" if format == "pickle" else cache_format.type

  if type == "custom":
    cache_format.dump(data, file)
  else:
    with open(file, "wb" if type == "binary" else "w") as f:
      cache_format.dump(data, f)

def cache(f, file, format="pickle"):
  if file.exists():
    return cache_read(file, format)

  res = f()
  cache_write(file, res, format)

  return res


_cache_env_default = dict(use_cache=True)
_cache_env = _cache_env_default


@contextlib.contextmanager
def cache_env(**kwargs):
  global _cache_env, _cache_env_default
  prev_cache_env = _cache_env
  try:
    _cache_env = fy.merge(_cache_env_default, kwargs)
    yield
  finally:
    _cache_env = prev_cache_env

def cached(dir_name=None, file_name="", format="pickle"):
  def cache_annotator(f):
    @fy.wraps(f)
    def cached_fn(*args, use_cache=None, **kwargs):
      if use_cache is None:
        use_cache = _cache_env["use_cache"]
      if not use_cache:
        return f(*args, **kwargs)

      fn = file_name(*args, **kwargs) if callable(file_name) else file_name
      fm = format(*args, **kwargs) if callable(format) else format
      ext = "json" if fm == "pretty_json" else fm

      dir = make_dir(dir_name)
      cache_file = dir / f"{fn}.{ext}"
      return cache(lambda: f(*args, **kwargs), cache_file, fm)
    return cached_fn
  return cache_annotator

class memoize:
  def __init__(self, f):
    self.f = f
    self.lut = {}

  def __call__(self, *args, **kwargs):
    key = tuple(args)

    if key in self.lut:
      return self.lut[key]

    res = self.f(*args, **kwargs)
    self.lut[key] = res
    return res

def walk_nested(f, state, s, **kwargs):
  "Walks nested structure recursively. Must not contain ref loops."
  state, filter = f(state, s, **kwargs)
  if filter is not False:
    if isinstance(filter, Iterable):
      for i in filter:
        if not isinstance(i, str) and isinstance(i, (Iterable, dict)):
          state = walk_nested(f, state, i, **kwargs)
        elif i in s:
          state = walk_nested(f, state, s[i], **kwargs)
    elif isinstance(s, dict):
      for v in s.values():
        state = walk_nested(f, state, v, **kwargs)
    elif not isinstance(s, str) and isinstance(s, Iterable):
      for v in s:
        state = walk_nested(f, state, v, **kwargs)
  return state
