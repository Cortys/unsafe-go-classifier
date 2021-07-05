import tensorflow as tf

def wl1_convolution(
  X_agg, X, ref_a, ref_b, directed=False, gathered=False):
  X_a = X if gathered else tf.gather(X, ref_a, axis=0)
  idx_b = tf.expand_dims(ref_b, axis=-1)
  X_agg = tf.tensor_scatter_nd_add(X_agg, idx_b, X_a)
  if not directed:
    X_b = X if gathered else tf.gather(X, ref_b, axis=0)
    idx_a = tf.expand_dims(ref_a, axis=-1)
    X_agg = tf.tensor_scatter_nd_add(X_agg, idx_a, X_b)
  return X_agg

def wl2_convolution(
  X, ref_a, ref_b, backref, backref_inv=None,
  combinator=lambda X_a, X_b: X_a + X_b):
  X_a = tf.gather(X, ref_a, axis=0)
  X_b = tf.gather(X, ref_b, axis=0)
  X_ab = combinator(X_a, X_b)
  backref = tf.expand_dims(backref, axis=-1)
  X_shape = tf.shape(X)
  X_agg = tf.scatter_nd(backref, X_ab, shape=X_shape)
  if backref_inv is not None:
    backref_inv = tf.expand_dims(backref_inv, axis=-1)
    X_ba = combinator(X_b, X_a)
    X_agg = tf.tensor_scatter_nd_add(X_agg, backref_inv, X_ba)
  return X_agg

def wl1_neighbor_norm(
  X, ref_a, ref_b, directed=False, selfloops=True,
  normalizer=tf.math.reciprocal_no_nan):
  X_size = tf.shape(X)[0]
  empty = tf.constant([], dtype=tf.float32)
  agg = tf.raw_ops.Bincount(arr=ref_b, size=X_size, weights=empty)
  if not directed:
    agg += tf.raw_ops.Bincount(arr=ref_a, size=X_size, weights=empty)
  if selfloops:
    agg += 1
  agg = tf.expand_dims(agg, -1)

  if normalizer is None:
    return agg
  return normalizer(agg)
