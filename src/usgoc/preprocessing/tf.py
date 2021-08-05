import tensorflow as tf
import tensorflow.keras as keras

import usgoc.preprocessing.graph.wl1 as wl1_enc
import usgoc.preprocessing.graph.wl2 as wl2_enc

def wl1(meta):
  node_dim, edge_dim = wl1_enc.feature_dims(**meta)
  graph_dim = meta.get("graph_feature_dim", 0)

  res = {
    "X": tf.TensorSpec(shape=[None, node_dim], dtype=tf.float32),
    "ref_a": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "ref_b": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "graph_idx": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "n": tf.TensorSpec(shape=[None], dtype=tf.int32)
  }

  if meta.get("multirefs", False):
    el = max(meta.get("edge_label_count", 1), 1)
    res["ref_a"] = (res["ref_a"],) * el
    res["ref_b"] = res["ref_a"]
  if edge_dim > 0:
    res["R"] = tf.TensorSpec(shape=[None, edge_dim], dtype=tf.float32)
  if meta.get("with_marked_node", False):
    res["marked_idx"] = tf.TensorSpec(shape=[None], dtype=tf.int32)
  if graph_dim > 0:
    res["graph_X"] = tf.TensorSpec(shape=[None, graph_dim], dtype=tf.float32)

  return res

def rwl1(meta):
  meta["with_ref_features"] = True
  return wl1(meta)

def mwl1(meta):
  meta["multirefs"] = True
  return wl1(meta)

def wl2(meta):
  X_dim = wl2_enc.feature_dim(**meta)
  graph_dim = meta.get("graph_feature_dim", 0)

  res = {
    "X": tf.TensorSpec(shape=[None, X_dim], dtype=tf.float32),
    "ref_a": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "ref_b": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "backref": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "graph_idx": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "n": tf.TensorSpec(shape=[None], dtype=tf.int32)
  }

  if meta.get("with_marked_node", False):
    res["marked_idx"] = tf.TensorSpec(shape=[None], dtype=tf.int32)
  if graph_dim > 0:
    res["graph_X"] = tf.TensorSpec(shape=[None, graph_dim], dtype=tf.float32)

  return res

def node_set(meta):
  node_dim, _ = wl1_enc.feature_dims(**meta)
  graph_dim = meta.get("graph_feature_dim", 0)

  res = {
    "X": tf.TensorSpec(shape=[None, node_dim], dtype=tf.float32),
    "graph_idx": tf.TensorSpec(shape=[None], dtype=tf.int32),
    "n": tf.TensorSpec(shape=[None], dtype=tf.int32)
  }

  if meta.get("with_marked_node", False):
    res["marked_idx"] = tf.TensorSpec(shape=[None], dtype=tf.int32)
  if graph_dim > 0:
    res["graph_X"] = tf.TensorSpec(shape=[None, graph_dim], dtype=tf.float32)

  return res

def vec32(meta):
  shape = [None, meta["feature_dim"]] if "feature_dim" in meta else [None]

  return tf.TensorSpec(shape=shape, dtype=tf.float32)

def int32(meta):
  return tf.TensorSpec(shape=[None], dtype=tf.int32)

def multiclass(meta):
  if "class_count" in meta:
    class_count = meta["class_count"]
  elif "max" in meta:
    class_count = 1 + meta["max"] - meta.get("min", 0)
  else:
    class_count = 2
  return vec32(dict(feature_dim=class_count))

def tup(enc, n=2):
  def tuple_enc(meta):
    signature = enc(meta)
    return (signature,) * n
  return tuple_enc


encodings = dict(
  null=lambda meta: tf.TensorSpec(shape=[1], dtype=tf.int32),
  int32=int32,
  int32_pair=tup(int32),
  wl1=wl1,
  rwl1=rwl1,
  mwl1=mwl1,
  wl2=wl2,
  node_set=node_set,
  float=vec32,
  vector=vec32,
  binary=vec32,
  multiclass=multiclass
)
encodings_with_tuples = dict(
  mwl1=["ref_a", "ref_b"]
)


def call_encoder(enc, meta, full_meta):
  encoder = encodings[enc]

  if getattr(encoder, "full_meta", False):
    return encoder(*full_meta)
  else:
    return encoder(meta)

def make_dataset(
  batch_generator, in_enc, in_meta=None, out_enc=None, out_meta=None,
  lazy_batching=True):
  full_meta = in_enc, in_meta, out_enc, out_meta
  input_sig = call_encoder(in_enc, in_meta, full_meta)

  if in_enc in encodings_with_tuples:
    for k in encodings_with_tuples[in_enc]:
      t = input_sig[k]
      if isinstance(t, tuple):
        del input_sig[k]
        for j, s in enumerate(t):
          input_sig[f"{k}_{j}"] = s

  if out_enc is None:
    signature = input_sig
  else:
    output_sig = call_encoder(out_enc, out_meta, full_meta)
    signature = (input_sig, output_sig)

  if lazy_batching:
    gen = batch_generator
  else:
    batches = list(batch_generator())
    gen = lambda: batches

  return tf.data.Dataset.from_generator(
    gen, output_signature=signature)

def _rec_make_inputs(spec, prefix=None):
  if isinstance(spec, tf.TensorSpec):
    if prefix is None:
      prefix = "X"
    return keras.Input(
      name=prefix, dtype=spec.dtype, shape=tuple(spec.shape.as_list()[1:]))

  if prefix is None:
    prefix = ""
  else:
    prefix = f"{prefix}_"

  if isinstance(spec, dict):
    return {
      k: _rec_make_inputs(s, prefix=f"{prefix}{k}")
      for k, s in spec.items()}
  elif isinstance(spec, tuple):
    return tuple(
      _rec_make_inputs(s, prefix=f"{prefix}{i}")
      for i, s in enumerate(spec))
  else:
    raise Exception(f"Unknown spec structure: {spec}")

def make_inputs(enc, meta={}):
  spec = encodings[enc](meta)
  return _rec_make_inputs(spec)

def tspec_to_ispec(spec, name="X"):
  return keras.layers.InputSpec(
    dtype=spec.dtype, shape=spec.shape.as_list(),
    name=name, allow_last_axis_squeeze=True)

def make_input_specs(enc, meta={}, type="input"):
  spec = encodings[enc](meta)
  if type == "input":
    convert = tspec_to_ispec
  else:
    def convert(spec, name="X"):
      return spec

  if enc in encodings_with_tuples:
    for k in encodings_with_tuples[enc]:
      t = spec.pop(k)
      for j, s in enumerate(t):
        spec[f"{k}_{j}"] = s

  if isinstance(spec, dict):
    return {
      n: convert(s, n)
      for n, s in spec.items()}

  if isinstance(spec, tuple):
    return tuple(convert(s, f"X_{i}") for i, s in enumerate(spec))

  return convert(spec)
