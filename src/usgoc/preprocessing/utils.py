import numpy as np

def make_graph_batch(
  encoded_graphs, ref_keys, flatten_multirefs=False,
  discard_empty=True):
  assert len(encoded_graphs) > 0, "Cannot create empty graph batch."
  X_batch_size = 0
  X_offset = 0
  batch_size = len(encoded_graphs)

  e0 = encoded_graphs[0]
  grouped = e0.get("group_idx", None) is not None
  with_marked_node = e0.get("marked_idx", None) is not None
  e0gf = e0.get("graph_X", None)
  with_graph_features = e0gf is not None
  if with_graph_features:
    graph_X_dim = len(e0gf)

  if ref_keys is not None and len(ref_keys) > 0:  # With refs:
    ref_a_key = ref_keys[0]
    ra0 = e0[ref_a_key]
    multirefs = isinstance(ra0, tuple)

    if multirefs:
      edge_features = False
      refs_count = len(ra0)
      ref_len = (lambda e: e["ref_sizes"][-1]) if refs_count > 1 \
          else (lambda e: len(e[ref_a_key][0]))
      ref_batch_size = np.zeros(refs_count, dtype=np.int32)
      ref_offset = np.zeros(refs_count, dtype=np.int32)
      for e in encoded_graphs:
        X_size, X_dim = e["X"].shape
        X_batch_size += X_size
        ref_batch_size += ref_len(e)
        if discard_empty and X_size == 0:
          batch_size -= 1

      refs_batch = {
        ref_key: tuple(
          np.empty(ref_batch_size[i], dtype=np.int32)
          for i in range(refs_count))
        for ref_key in ref_keys}
    else:
      ref_len = lambda e: len(e[ref_a_key])
      refs_count = 1
      ref_batch_size = 0
      ref_offset = 0
      for e in encoded_graphs:
        X_size, X_dim = e["X"].shape
        X_batch_size += X_size
        ref_batch_size += len(e[ref_a_key])
        if discard_empty and X_size == 0:
          batch_size -= 1

      refs_batch = {
        ref_key: np.empty(ref_batch_size, dtype=np.int32)
        for ref_key in ref_keys}

      edge_features = "R" in e0 and e0["R"] is not None

      if edge_features:
        R_dim = e0["R"].shape[1]
        R_batch = np.empty((ref_batch_size, R_dim), dtype=np.float32)
  else:  # No refs:
    multirefs = False
    edge_features = False
    ref_keys = []
    ref_offset = 0
    ref_len = lambda e: 0
    refs_batch = {}
    for e in encoded_graphs:
      X_size, X_dim = e["X"].shape
      X_batch_size += X_size
      if discard_empty and X_size == 0:
        batch_size -= 1

  X_batch = np.empty((X_batch_size, X_dim), dtype=np.float32)
  n_batch = np.empty(batch_size, dtype=np.int32)
  graph_idx = np.empty(X_batch_size, dtype=np.int32)
  if grouped:
    group_idx_batch = np.empty(batch_size, dtype=np.int32)
  if with_marked_node:
    marked_idx_batch = np.empty(batch_size, dtype=np.int32)
  if with_graph_features:
    graph_X_batch = np.empty((batch_size, graph_X_dim), dtype=np.float32)

  i = 0

  for e in encoded_graphs:
    X = e["X"]
    X_size = len(X)

    if discard_empty and X_size == 0:  # discard empty graphs
      continue

    next_X_offset = X_offset + X_size
    next_ref_offset = ref_offset + ref_len(e)

    X_batch[X_offset:next_X_offset, :X_dim] = X
    graph_idx[X_offset:next_X_offset] = i
    n_batch[i] = e["n"]

    if grouped:
      group_idx_batch[i] = e["group_idx"]

    if with_marked_node:
      marked_idx = e["marked_idx"]
      if marked_idx != -1:
        marked_idx += X_offset
      marked_idx_batch[i] = marked_idx

    if with_graph_features:
      graph_X_batch[i] = e["graph_X"]

    if multirefs:
      for ref_key in ref_keys:
        rbk = refs_batch[ref_key]
        ek = e[ref_key]
        for j in range(refs_count):
          rbk[j][ref_offset[j]:next_ref_offset[j]] = ek[j] + X_offset
    else:
      for ref_key in ref_keys:
        refs_batch[ref_key][ref_offset:next_ref_offset] = e[ref_key] + X_offset

      if edge_features:
        R_batch[ref_offset:next_ref_offset] = e["R"]

    X_offset = next_X_offset
    ref_offset = next_ref_offset
    i += 1

  if multirefs and flatten_multirefs:
    tuple_refs_batch = refs_batch
    refs_batch = {
      f"{k}_{j}": t[j]
      for k, t in tuple_refs_batch.items()
      for j in range(refs_count)
    }

  if edge_features:
    refs_batch["R"] = R_batch
  if grouped:
    refs_batch["group_idx"] = group_idx_batch
  if with_marked_node:
    refs_batch["marked_idx"] = marked_idx_batch
  if with_graph_features:
    refs_batch["graph_X"] = graph_X_batch

  return {
    "X": X_batch,
    "n": n_batch,
    **refs_batch,
    "graph_idx": graph_idx,
  }
