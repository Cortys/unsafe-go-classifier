import tensorflow as tf
import numpy as np
import funcy as fy
import matplotlib
import matplotlib.pyplot as plt

# import usgoc.preprocessing.graph.wl1 as wl1
import usgoc.datasets.unsafe_go as dataset
import usgoc.models.gnn as gnn
import usgoc.utils as utils
import usgoc.metrics.multi as mm

with utils.cache_env(use_cache=True):
  files = dataset.load_filenames()
  ds = dataset.load_dataset()
  graphs, targets = ds

with utils.cache_env(use_cache=True):
  # dims = dataset.create_graph_dims(graphs)
  splits = dataset.get_split_idxs(ds)
  labels1, labels2 = dataset.create_target_label_dims(ds)
  labels1_keys = labels1.keys()
  labels2_keys = labels2.keys()
  labels1_inv = fy.flip(labels1)
  labels2_inv = fy.flip(labels2)

# model = gnn.MLP
model1 = gnn.DeepSets
# model = gnn.GCN
# model = gnn.GIN
model = gnn.GGNN
# model = gnn.RGCN
# model = gnn.RGIN

fold = 0

with utils.cache_env(use_cache=True):
  dims, train_ds, val_ds, test_ds = dataset.get_encoded_dataset_slices(
    ds, model.in_enc, splits, fold, limit_id="v127_d127_f127_p127",
    batch_size_limit=200)
  train_ds = train_ds.cache()
  val_ds = val_ds.cache()
  train_slice, val_slice, test_slice = dataset.get_dataset_slices(
    ds, splits, fold)

def experiment(model):
  m = model(
    node_label_count=dims["node_label_count"],
    conv_directed=True,
    conv_layer_units=[64] * 5, fc_layer_units=[64] * 5,
    conv_activation="relu",
    conv_inner_activation="relu",
    fc_activation="relu",
    out_activation=None,
    pooling="sum", learning_rate=0.001)

  m.fit(train_ds, validation_data=val_ds, verbose=2, epochs=500)
  res = m.evaluate(test_ds, return_dict=True)
  print(res)
  return m


m = experiment(model1)
m2 = experiment(model)

def debug_graph(i=None, file=None, test_j=None, val_j=None, draw=True):
  if i is None:
    if file is not None:
      i = fy.first(fy.filter(
        lambda e: "e6a3feac3fc40c305816" in e[1], enumerate(files)))[0]
    elif test_j is not None:
      i = splits[fold]["test"][test_j]
    elif val_j is not None:
      i = splits[fold]["validation"][val_j]
  print(graphs[i].source_code)
  print(files[i], labels1_inv[targets[0][i]], labels2_inv[targets[1][i]])
  if draw:
    utils.draw_graph(graphs[i], edge_colors=True, layout="dot")
  return i

def get_singleton_ds(i):
  with utils.cache_env(use_cache=False):
    return dataset.dataset_encoders[model.in_enc](dataset.slice(ds, [i]), dims)

def preds_to_dicts(preds, j):
  prob1 = np.around(tf.nn.softmax(preds[0], -1).numpy()[j], 3)
  prob2 = np.around(tf.nn.softmax(preds[1], -1).numpy()[j], 3)
  d1 = fy.zipdict(labels1_keys, zip(list(prob1), list(preds[0][j])))
  d2 = fy.zipdict(labels2_keys, zip(list(prob2), list(preds[1][j])))
  return d1, d2

def draw_confusion(pred_labels, target_labels, normalize=True):
  m1 = mm.sparse_multi_confusion_matrix(
    tf.constant(target_labels[0], dtype=tf.int32),
    tf.constant(pred_labels[0], dtype=tf.int32)).numpy()
  m2 = mm.sparse_multi_confusion_matrix(
    tf.constant(target_labels[1], dtype=tf.int32),
    tf.constant(pred_labels[1], dtype=tf.int32)).numpy()
  a1 = np.sum(np.diag(m1)) / np.sum(m1)
  a2 = np.sum(np.diag(m2)) / np.sum(m2)
  print("L1-Acc:", a1, "L2-Acc:", a2)
  if normalize:
    m1 = np.around(m1 / np.sum(m1, axis=1, keepdims=True), 2) * 100
    m2 = np.around(m2 / np.sum(m2, axis=1, keepdims=True), 2) * 100
  utils.draw_confusion_matrix(m1.astype(int), labels1_keys)
  utils.draw_confusion_matrix(m2.astype(int), labels2_keys)


# interesting i's: 40, 70, 874 (44b41ab329d2624a449e),
# Instances on which both DeepSets and GGNN fail (both labels):
# - 43 (a793d8b779605120ec52) true: cast-basic ffi (short snippet)
# - 79 (836d21cfe55279cd8e46) true: delegate atomic (long snippet)
# - 1236 (27b69b7c042b73cf9275) true: memory-access layout (short/medium snippet)

# Instances on wihich only GGNN fails (but not DeepSets):
# - 84 (b01a05c45ecc95b792c8) true: cast-header types (short/medium)
#   Code:
#   func bytesHash(b []byte, seed uintptr) uintptr {
#	    s := (*slice)(unsafe.Pointer(&b))
#	    return memhash(s.array, seed, uintptr(s.len))
#   }
# - 1015 (39edbcf5d9443fc73e3a) true: pointer-arithmetic serialization (short)
# - 1313 (b54f66ea4c799f26617d) true: cast-struct generics (short)
#   Code: func (h *Header) Int8s() []int8 { return *(*[]int8)(unsafe.Pointer(h)) }
#   DeepSets Preds:
#   ({'cast-basic': (0.544, 13.852097),
#     'cast-bytes': (0.0, -0.26986185),
#     'cast-header': (0.0, -3.9276118),
#     'cast-pointer': (0.455, 13.672801),
#     'cast-struct': (0.001, 7.118796),
#     'definition': (0.0, -37.71391),
#     'delegate': (0.0, -3.9902112),
#     'memory-access': (0.0, -4.7687745),
#     'pointer-arithmetic': (0.0, -17.577158),
#     'syscall': (0.0, -22.886244),
#     'unused': (0.0, -5.127344)},
#    {'atomic': (0.0, -8.450554),
#     'efficiency': (0.001, 3.8338575),
#     'ffi': (0.0, 0.4648861),
#     'generics': (0.994, 10.651982),
#     'hide-escape': (0.0, -19.393679),
#     'layout': (0.0, -3.1287308),
#     'no-gc': (0.0, -17.564428),
#     'reflect': (0.0, 1.974027),
#     'serialization': (0.0, 0.13683213),
#     'types': (0.004, 5.250824),
#     'unused': (0.0, -0.42025906)})
#   GGNN Preds:
#   ({'cast-basic': (0.0, 1.3986396),
#     'cast-bytes': (0.993, 19.658085),
#     'cast-header': (0.0, -14.408668),
#     'cast-pointer': (0.0, 11.910549),
#     'cast-struct': (0.0, 6.641031),
#     'definition': (0.0, -29.842493),
#     'delegate': (0.006, 14.610508),
#     'memory-access': (0.0, 5.646147),
#     'pointer-arithmetic': (0.0, -32.213),
#     'syscall': (0.0, -41.637238),
#     'unused': (0.0, -21.783506)},
#    {'atomic': (0.0, 10.781407),
#     'efficiency': (0.007, 13.999677),
#     'ffi': (0.0, -11.787495),
#     'generics': (0.0, 11.160701),
#     'hide-escape': (0.0, -27.821772),
#     'layout': (0.0, -1.9805977),
#     'no-gc': (0.0, -40.828663),
#     'reflect': (0.0, 3.5758069),
#     'serialization': (0.993, 18.984678),
#     'types': (0.0, -6.821523),
#     'unused': (0.0, -30.649702)})
# - 1393 (a4c0182265f30feaa1f2) true: cast-bytes efficiency
#   Code: func (m *SortedMap) Get(key string) (value interface{}, ok bool) {
#	          return m.trie.Get(*(*[]byte)(unsafe.Pointer(&key)))
#         }

# NOTE: On first, third and last two examples serialization was wrongly predicted by GNN (instead of generics/efficiency). Why? DeepSet correctly detects generic, but not efficiency

train_pred = m.predict(train_ds)
train_pred2 = m2.predict(train_ds)
s_pred = m.predict(test_ds)
s_pred2 = m2.predict(test_ds)
s_pred_labels = tf.cast(tf.stack([tf.argmax(s_pred[0], -1), tf.argmax(s_pred[1], -1)], 1), tf.int32)
s_pred_labels2 = tf.cast(tf.stack([tf.argmax(s_pred2[0], -1), tf.argmax(s_pred2[1], -1)], 1), tf.int32)
target_labels = tf.stack(list(test_ds)[0][1], 1)
s_pred
draw_confusion(train_pred, train_slice[1], True)
draw_confusion(train_pred2, train_slice[1], True)
draw_confusion(s_pred, test_slice[1], True)
draw_confusion(s_pred2, test_slice[1], True)

pred_matches = (s_pred_labels == target_labels).numpy()
pred_matches2 = (s_pred_labels2 == target_labels).numpy()
problem_js = np.where(~(pred_matches[:, 0] | pred_matches[:, 1]))[0]
problem_js2 = np.where(~(pred_matches2[:, 0] | pred_matches2[:, 1]))[0]
set(problem_js) & set(problem_js2)
set(problem_js2) - set(problem_js)
len(problem_js2)

# j = problem_js[4]
j = 32; i = None
# i = 84
i = debug_graph(i=i, test_j=j, draw=True)
i
preds_to_dicts(s_pred, j)
preds_to_dicts(s_pred2, j)
s_pred_labels2[j]
