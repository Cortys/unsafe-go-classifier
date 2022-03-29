import numpy as np

def sets_to_vecs(sets, max_size=11):
  set_vecs = np.zeros((len(sets), max_size), dtype=np.int32)
  for i, s in enumerate(sets):
    set_vecs[i, s] = 1
  return set_vecs

def cooccurrence_matrix(vecs):
  dims = vecs.shape[-1]
  mat = np.zeros((dims, dims), dtype=np.int32)
  for i in range(dims):
    mat[i] = np.sum(vecs[(vecs[:,i] == 1)], axis=0)
  return mat

def vec_size_buckets(vecs, cummulative=True):
  dims = vecs.shape[-1]
  sizes = np.sum(vecs, -1)
  if cummulative:
    op=lambda s, i: s >= i
  else:
    op=lambda s, i: s == i
  return [vecs[op(sizes, i + 1)] for i in range(dims)]

def sets_to_cooccurrence(sets, max_size=11):
  return cooccurrence_matrix(sets_to_vecs(sets, max_size))
