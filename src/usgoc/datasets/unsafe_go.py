import glob
import json
import networkx as nx
from pathlib import Path

import usgoc.utils as utils

RAW_DATASET_PATTERN = "/app/raw/unsafe-go-dataset/**/*.json"
DATA_DIR = Path("/app/data/unsafe-go-dataset")

def load_raw():
  files = glob.glob(RAW_DATASET_PATTERN, recursive=True)
  res = []
  for f in files:
    with open(f, "r") as fp:
      res.append(json.load(fp))
  return res

@utils.cached(DATA_DIR, "labels", "pretty_json")
def collect_labels(raw_dataset):
  label1s = set()
  label2s = set()
  for inst in raw_dataset:
    usage = inst["usage"]
    label1s.add(usage["label1"])
    label2s.add(usage["label2"])
  label1s = sorted(label1s)
  label2s = sorted(label2s)
  label1s = dict(zip(label1s, range(len(label1s))))
  label2s = dict(zip(label2s, range(len(label2s))))
  return label1s, label2s

def usage_to_target(usage, labels):
  label1s, label2s = labels
  return label1s[usage["label1"]], label2s[usage["label2"]]

def cfg_to_graph(cfg):
  g = nx.DiGraph()

  return g

@utils.cached(DATA_DIR, "in_nx")
def raw_to_graphs(raw_dataset):
  return [cfg_to_graph(inst["cfg"]) for inst in raw_dataset]

@utils.cached(DATA_DIR, "out_ids")
def raw_to_usages(raw_dataset):
  labels = collect_labels(raw_dataset)
  return [usage_to_target(inst["usage"], labels) for inst in raw_dataset]

def load_dataset():
  raw_dataset = load_raw()
  return raw_to_graphs(raw_dataset), raw_to_usages(raw_dataset)
