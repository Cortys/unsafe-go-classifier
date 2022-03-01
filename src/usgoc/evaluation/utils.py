from typing import List
import mlflow

import usgoc.utils as utils

mlflow.set_tracking_uri(f"file:{utils.PROJECT_ROOT}/mlruns")

FOLDS_MAX = 10
REPEATS_MAX = 3


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
  eid = mlflow.tracking.fluent._get_experiment_id()
  parent_id = mlflow.active_run().info.run_id
  max_results = 1
  conditions = [
    "tags.type = 'inner'",
    f"tags.mlflow.parentRunId = '{parent_id}'"]
  if fold is not None:
    conditions.append(f"tags.fold = '{fold}'")
  else:
    max_results = FOLDS_MAX
  if repeat is not None:
    conditions.append(f"tags.repeat = '{repeat}'")
  else:
    max_results *= REPEATS_MAX
  query = " and ".join(conditions)
  runs = mlflow.search_runs(
    [eid], query, max_results=max_results, output_format="list")

  return runs

def find_inner_run(fold, repeat) -> mlflow.entities.Run | None:
  runs = find_inner_runs(fold, repeat)
  if len(runs) == 0:
    return None
  return runs[0]
