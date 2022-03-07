import usgoc.utils as utils
import usgoc.datasets.unsafe_go as dataset

dataset_names = ["usgo_v1"]
convert_modes = dataset.convert_modes
limit_ids = dataset.get_dim_limit_dict().keys()
evaluate_convert_modes = [
  "atomic_blocks",
  # "split_blocks"
]
evaluate_limit_ids = [
  "v127_d127_f127_p127",
  "v0_d127_f127_p127",
  "v127_d0_f127_p127",
  "v127_d127_f0_p127",
  "v127_d127_f127_p0",
  "v0_d0_f0_p0",
  # "v127_d127_f127_p127_core",
]

def get_batch_size(convert_mode=None, limit_id=None):
  if convert_mode == "split_blocks":
    return 100
  return 200

def get_dims(
  fold=0, name=dataset_names[0], convert_mode=None, limit_id=None):
  return dataset.create_graph_dims(
    None, limit_id=limit_id, split_id=f"{fold}_0_train", mode=convert_mode,
    force_cache=True)

def get_target_label_dims():
  return dataset.create_target_label_dims(None, force_cache=True)

def get_encoded(
  in_enc, fold=0,
  name=dataset_names[0], convert_mode=None, limit_id=None,
  batch_size_limit=None):
  if batch_size_limit is None:
    batch_size_limit = get_batch_size(convert_mode, limit_id)

  ds = dataset.load_dataset(mode=convert_mode)
  splits = dataset.get_split_idxs(ds)
  dims, train_ds, val_ds, test_ds = dataset.get_encoded_dataset_slices(
    ds, in_enc, splits, fold, mode=convert_mode, limit_id=limit_id,
    batch_size_limit=batch_size_limit)
  train_ds = train_ds.cache()
  val_ds = val_ds.cache()

  return dims, train_ds, val_ds, test_ds
