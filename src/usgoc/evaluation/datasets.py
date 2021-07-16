import usgoc.datasets.unsafe_go as dataset

dataset_names = ["usgo_v1"]
evaluate_limit_ids = [
  "v127_d127_f127_p127"
]

def get_encoded(
  in_enc, fold=0,
  name=dataset_names[0], limit_id=None,
  batch_size_limit=200):
  ds = dataset.load_dataset()
  splits = dataset.get_split_idxs(ds)
  dims, train_ds, val_ds, test_ds = dataset.get_encoded_dataset_slices(
    ds, in_enc, splits, fold, limit_id=limit_id,
    batch_size_limit=batch_size_limit)
  train_ds = train_ds.cache()
  val_ds = val_ds.cache()

  return dims, train_ds, val_ds, test_ds
