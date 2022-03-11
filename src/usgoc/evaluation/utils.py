from __future__ import annotations
from typing import List
import mlflow

import usgoc.utils as utils

mlflow.set_tracking_uri(f"file:{utils.PROJECT_ROOT}/mlruns")

FOLDS_MAX = 10
REPEATS_MAX = 3

_inner_runs_cache = None

def find_outer_run(convert_mode, limit_id, model_name) -> mlflow.entities.Run | None:
  eid = mlflow.tracking.fluent._get_experiment_id()
  runs = mlflow.search_runs(
    [eid],
    " and ".join([
      "tags.type = 'outer'",
      f"tags.convert_mode = '{convert_mode}'",
      f"tags.limit_id = '{limit_id}'",
      f"tags.model = '{model_name}'"]),
    max_results=1, output_format="list")
  if len(runs) == 0:
    return None

  return runs[0]

def find_inner_runs(fold=None, repeat=None) -> List[mlflow.entities.Run]:
  global _inner_runs_cache
  eid = mlflow.tracking.fluent._get_experiment_id()
  parent_id = mlflow.active_run().info.run_id

  if _inner_runs_cache is not None and _inner_runs_cache[0] == (eid, parent_id):
    runs = _inner_runs_cache[1]
  else:
    max_results = FOLDS_MAX * REPEATS_MAX
    conditions = [
      "tags.type = 'inner'",
      f"tags.mlflow.parentRunId = '{parent_id}'"]
    query = " and ".join(conditions)
    runs = mlflow.search_runs(
      [eid], query, max_results=max_results, output_format="list")
    _inner_runs_cache = ((eid, parent_id), runs)

  if fold is None and repeat is None:
    return runs

  filtered = [
    run
    for run in runs
    if (fold is None or int(run.data.tags["fold"]) == fold) \
    and (repeat is None or int(run.data.tags["repeat"]) == repeat)]

  return filtered

def invalidate_inner_run_cache():
  global _inner_runs_cache
  _inner_runs_cache = None

def find_inner_run(fold, repeat) -> mlflow.entities.Run | None:
  runs = find_inner_runs(fold, repeat)
  if len(runs) == 0:
    return None
  if len(runs) > 1:
    print(f"WARNING: Found multiple runs for {fold}, {repeat}:")
    for run in runs:
      print(f"- {run.info.run_id}")
  return runs[0]

def find_orphaned_runs():
  eid = mlflow.tracking.fluent._get_experiment_id()
  outer_runs = mlflow.search_runs(
    [eid], "tags.type = 'outer'",
    output_format="list")
  orphan_runs = mlflow.search_runs(
    [eid],
    " and ".join(["tags.type = 'inner'"] + [
      f"tags.mlflow.parentRunId != '{run.info.run_id}'"
      for run in outer_runs
    ]),
    output_format="list")
  return orphan_runs
