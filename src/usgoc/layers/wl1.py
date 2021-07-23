import tensorflow as tf
from tensorflow import keras

from usgoc.layers.utils import wl1_convolution, wl2_convolution, \
  wl1_neighbor_norm

class GCNPreprocessLayer(keras.layers.Layer):
  def __init__(self):
    super().__init__()

  def call(self, input):
    X = input["X"]
    ref_a = input["ref_a"]
    ref_b = input["ref_b"]
    norm = wl1_neighbor_norm(X, ref_a, ref_b, normalizer=tf.math.rsqrt)

    return {**input, "norm": norm}

class RGCNPreprocessLayer(keras.layers.Layer):
  def __init__(
    self, directed=False, mode="crossrel", selfloops=True, reverse=False):
    super().__init__()
    self.directed = directed
    assert mode in {None, "rel", "crossrel"}
    self.mode = mode
    self.selfloops = selfloops
    self.reverse = reverse

  def get_config(self):
    return dict(
      **super().get_config(),
      directed=self.directed,
      mode=self.mode,
      selfloops=self.selfloops,
      reverse=self.reverse)

  def call(self, input):
    mode = self.mode
    if not mode:
      return input

    X = input["X"]
    ref_a = input["ref_a"]
    ref_b = input["ref_b"]

    normalizer = None if mode == "crossrel" else tf.math.reciprocal_no_nan
    norms = tuple(
      wl1_neighbor_norm(
        X, ra, rb, directed=self.directed, selfloops=self.selfloops,
        normalizer=normalizer)
      for ra, rb in zip(ref_a, ref_b))

    if mode == "crossrel":
      norms = tf.math.accumulate_n(norms)
      norms = tf.math.reciprocal_no_nan(norms)

    if self.directed and self.reverse:
      rev_norms = tuple(
        wl1_neighbor_norm(
          X, rb, ra, directed=self.directed, selfloops=self.selfloops,
          normalizer=normalizer)
        for ra, rb in zip(ref_a, ref_b))

      if mode == "crossrel":
        rev_norms = tf.math.accumulate_n(rev_norms)
        rev_norms = tf.math.reciprocal_no_nan(rev_norms)

      return {**input, "norms": norms, "rev_norms": rev_norms}

    return {**input, "norms": norms}

class DeepSetsLayer(keras.layers.Layer):
  def __init__(self, units, use_bias=True, activation=None):
    super().__init__()
    self.units = units
    self.use_bias = use_bias
    self.activation = keras.activations.get(activation)

  def get_config(self):
    return dict(
      **super().get_config(),
      units=self.units,
      use_bias=self.use_bias,
      activation=keras.activations.serialize(self.activation))

  def build(self, input_shape):
    super().build(input_shape)
    X_shape = input_shape["X"]
    vert_dim = X_shape[-1]

    self.W = self.add_weight(
      "W", shape=(vert_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.use_bias:
      self.b = self.add_weight(
        "b", shape=(self.units,),
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    X = input["X"]
    X_out = X @ self.W
    if self.use_bias:
      X_out = tf.nn.bias_add(X_out, self.b)
    X_out = self.activation(X_out)

    return {**input, "X": X_out}

class GCNLayer(keras.layers.Layer):
  def __init__(self, units, use_bias=True, activation=None):
    super().__init__()
    self.units = units
    self.use_bias = use_bias
    self.activation = keras.activations.get(activation)

  def get_config(self):
    return dict(
      **super().get_config(),
      units=self.units,
      use_bias=self.use_bias,
      activation=keras.activations.serialize(self.activation))

  def build(self, input_shape):
    super().build(input_shape)
    X_shape = input_shape["X"]
    vert_dim = X_shape[-1]

    self.W = self.add_weight(
      "W", shape=(vert_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.use_bias:
      self.b = self.add_weight(
        "b", shape=(self.units,),
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    X = input["X"]
    ref_a = input["ref_a"]
    ref_b = input["ref_b"]
    norm = input["norm"]
    X_norm = X * norm
    X_agg = wl1_convolution(X_norm, X_norm, ref_a, ref_b)
    X_aggnorm = X_agg * norm

    X_out = X_aggnorm @ self.W
    if self.use_bias:
      X_out = tf.nn.bias_add(X_out, self.b)
    X_out = self.activation(X_out)

    return {**input, "X": X_out}

class RGNNLayer(keras.layers.Layer):
  def __init__(
    self, units, use_bias=True,
    activation=None, inner_activation=None,
    directed=False, selfloops=True, reverse=False,
    with_final_dense=False):
    super().__init__()
    self.units = units
    self.use_bias = use_bias
    self.activation = keras.activations.get(activation)
    self.inner_activation = keras.activations.get(inner_activation)
    self.directed = directed
    self.selfloops = selfloops
    self.reverse = directed and reverse
    self.with_final_dense = with_final_dense

  def get_config(self):
    return dict(
      **super().get_config(),
      units=self.units,
      use_bias=self.use_bias,
      activation=keras.activations.serialize(self.activation),
      inner_activation=keras.activations.serialize(self.inner_activation),
      directed=self.directed,
      selfloops=self.selfloops,
      reverse=self.reverse,
      with_final_dense=self.with_final_dense)

  def build(self, input_shape):
    super().build(input_shape)
    X_shape = input_shape["X"]
    vert_dim = X_shape[-1]
    ref_count = len(input_shape["ref_a"])

    if self.selfloops:
      self.W_0 = self.add_weight(
        "W_0", shape=(vert_dim, self.units),
        trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W = self.add_weight(
      "W", shape=(ref_count, vert_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.with_final_dense:
      self.W_fin = self.add_weight(
        "W_fin", shape=(self.units, self.units),
        trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.reverse:
      self.W_rev = self.add_weight(
        "W_rev", shape=(ref_count, vert_dim, self.units),
        trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.use_bias:
      self.b = self.add_weight(
        "b", shape=(self.units,),
        trainable=True, initializer=tf.initializers.Zeros)
      if self.with_final_dense:
        self.b_fin = self.add_weight(
          "b_fin", shape=(self.units,),
          trainable=True, initializer=tf.initializers.Zeros)

  def _convolve(self, X, ref_a, ref_b, W, norms):
    ref_count = len(ref_a)
    X_zero = tf.zeros_like(X)
    X_aggs = [None] * ref_count

    for i in range(ref_count):
      X_agg = wl1_convolution(
        X_zero, X, ref_a[i], ref_b[i], directed=self.directed)
      X_aggs[i] = (X_agg @ W[i, :, :])
      if norms is not None:
        X_aggs[i] *= norms[i] if isinstance(norms, tuple) else norms

    return X_aggs

  def call(self, input):
    X = input["X"]
    ref_a = input["ref_a"]
    ref_b = input["ref_b"]
    norms = input.get("norms", None)
    X_aggs = self._convolve(X, ref_a, ref_b, self.W, norms)
    if self.reverse:
      rev_norms = input.get("rev_norms", None)
      X_aggs += self._convolve(X, ref_b, ref_a, self.W_rev, rev_norms)
    else:
      rev_norms = None

    if self.selfloops:
      X_self = X @ self.W_0
      if isinstance(norms, tuple):
        X_self *= tf.math.accumulate_n(norms)
      if norms is not None:
        if rev_norms is not None:
          X_self *= norms + rev_norms
        else:
          X_self *= norms
      X_aggs.append(X_self)

    X_out = tf.math.accumulate_n(X_aggs)
    if self.use_bias:
      X_out = tf.nn.bias_add(X_out, self.b)

    if self.with_final_dense:
      X_out = self.inner_activation(X_out)
      X_out = X_out @ self.W_fin
      if self.use_bias:
        X_out = tf.nn.bias_add(X_out, self.b_fin)

    X_out = self.activation(X_out)

    return {**input, "X": X_out}

class GINLayer(keras.layers.Layer):
  def __init__(
    self, units, use_bias=True, activation=None, inner_activation=None):
    super().__init__()
    self.units = units
    self.use_bias = use_bias
    self.activation = keras.activations.get(activation)
    self.inner_activation = keras.activations.get(inner_activation)

  def get_config(self):
    return dict(
      **super().get_config(),
      units=self.units,
      use_bias=self.use_bias,
      activation=keras.activations.serialize(self.activation),
      inner_activation=keras.activations.serialize(self.inner_activation))

  def build(self, input_shape):
    super().build(input_shape)
    X_shape = input_shape["X"]
    vert_dim = X_shape[-1]
    hidden_dim = self.units

    self.W_hidden = self.add_weight(
      "W_hidden", shape=(vert_dim, hidden_dim),
      trainable=True, initializer=tf.initializers.GlorotUniform)
    self.W_out = self.add_weight(
      "W_out", shape=(hidden_dim, self.units),
      trainable=True, initializer=tf.initializers.GlorotUniform)

    if self.use_bias:
      self.b_hidden = self.add_weight(
        "b_hidden", shape=(hidden_dim,),
        trainable=True, initializer=tf.initializers.Zeros)
      self.b_out = self.add_weight(
        "b_out", shape=(self.units,),
        trainable=True, initializer=tf.initializers.Zeros)

  def call(self, input):
    X = input["X"]
    ref_a = input["ref_a"]
    ref_b = input["ref_b"]
    X_agg = wl1_convolution(X, X, ref_a, ref_b)

    X_hid = X_agg @ self.W_hidden
    if self.use_bias:
      X_hid = tf.nn.bias_add(X_hid, self.b_hidden)
    X_hid = self.inner_activation(X_hid)
    X_out = X_hid @ self.W_out
    if self.use_bias:
      X_out = tf.nn.bias_add(X_out, self.b_out)
    X_out = self.activation(X_out)

    return {**input, "X": X_out}

class GGNNLayer(keras.layers.Layer):
  def __init__(
    self, units, use_bias=True, use_attention=False, use_diff=False,
    depth=2, directed=False, reverse=False,
    activation="tanh", recurrent_activation="sigmoid",
    inner_activation=None):
    super().__init__()
    self.units = units
    self.use_bias = use_bias
    self.use_attention = use_attention
    self.use_diff = use_diff
    self.depth = depth
    self.directed = directed
    self.reverse = directed and reverse
    self.activation = keras.activations.get(activation)
    self.recurrent_activation = keras.activations.get(recurrent_activation)
    self.inner_activation = keras.activations.get(inner_activation)

    self.gru = keras.layers.GRUCell(
      units=units, use_bias=use_bias,
      activation=self.activation,
      recurrent_activation=self.recurrent_activation)
    self.msg_net = keras.Sequential([
      keras.layers.Dense(
        units=units, use_bias=use_bias, activation=self.inner_activation)
      for _ in range(depth)], "msg_net")

    if self.reverse:
      self.rev_msg_net = keras.Sequential([
        keras.layers.Dense(
          units=units, use_bias=use_bias, activation=self.inner_activation)
        for _ in range(depth)], "rev_msg_net")

    if use_attention:
      self.att_net = keras.Sequential([
        keras.layers.Dense(
          units=units, use_bias=use_bias,
          activation="sigmoid" if i + 1 == depth else self.inner_activation)
        for i in range(depth)], "att_net")

  def get_config(self):
    return dict(
      **super().get_config(),
      units=self.units,
      use_bias=self.use_bias,
      use_attention=self.use_attention,
      use_diff=self.use_diff,
      depth=self.depth,
      directed=self.directed,
      reverse=self.reverse,
      activation=keras.activations.serialize(self.activation),
      recurrent_activation=keras.activations.serialize(
        self.recurrent_activation),
      inner_activation=keras.activations.serialize(self.inner_activation))

  def build(self, input_shape):
    super().build(input_shape)
    X_shape = input_shape["X"]
    vert_dim = X_shape[-1]
    vert_adapt = vert_dim != self.units
    self.vert_adapt = vert_adapt

    if vert_adapt:
      self.W_adapt = self.add_weight(
        "W_adapt", shape=(vert_dim, self.units),
        trainable=True, initializer=tf.initializers.GlorotUniform)

  def call(self, input):
    X = input["X"]
    ref_a = input["ref_a"]
    ref_b = input["ref_b"]

    if self.vert_adapt:
      X = X @ self.W_adapt

    if self.use_diff or self.use_attention:
      rev_ref = None if self.directed and not self.reverse else ref_a
      X_agg = wl2_convolution(
        X, ref_a, ref_b, ref_b, rev_ref,
        self.combine, self.combine_rev)
    else:
      X_msg = self.msg_net(X)
      X_zero = tf.zeros_like(X_msg)
      X_agg = wl1_convolution(
        X_zero, X_msg, ref_a, ref_b, directed=self.directed)
      if self.reverse:
        X_msg_rev = self.rev_msg_net(X)
        X_agg = wl1_convolution(
          X_agg, X_msg_rev, ref_b, ref_a, directed=True)
    X_out = self.gru(X_agg, X)[0]

    return {**input, "X": X_out}

  def combine(self, X_a, X_b):
    X_diff = X_a - X_b
    X_msg = self.msg_net(X_diff) if self.use_diff else self.msg_net(X_a)
    if self.use_attention:
      X_msg *= self.att_net(X_diff)
    return X_msg

  def combine_rev(self, X_b, X_a):
    if not self.reverse:
      return self.combine(X_b, X_a)
    X_diff = X_b - X_a
    if self.use_diff:
      X_msg = self.rev_msg_net(X_diff)
    else:
      X_msg = self.rev_msg_net(X_b)
    if self.use_attention:
      X_msg *= self.att_net(X_diff)
    return X_msg
