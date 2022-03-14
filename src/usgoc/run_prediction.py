import os
import json
import yaml
import click
import subprocess
import funcy as fy
import networkx as nx
import tensorflow as tf

import usgoc.utils as utils
import usgoc.datasets.unsafe_go as dataset
import usgoc.metrics.multi as mm
import usgoc.postprocessing.conformal as conf

PROJECTS_DIR = "/projects"
EXPORT_DIR = f"{utils.PROJECT_ROOT}/exported_models"
GO_VERSION = "1.14.3"

def get_cfg_json(
  base, project, package, file, line, snippet="",
  dist=0, cache_dist=3, go_version=GO_VERSION, **kwargs):
  go_path = subprocess.run(
    [f"go{go_version}", "env", "GOROOT"],
    stdout=subprocess.PIPE, check=True, encoding="utf-8").stdout.strip()
  dac_env = os.environ.copy()
  dac_env["GOPATH"] = "/root/go"
  dac_env["PATH"] = f"{go_path}/bin:{dac_env['PATH']}"
  p = subprocess.run(
    ["/usr/bin/data-acquisition-tool", "cfg",
     "--base", base,
     "--project", project,
     "--package", package,
     "--file", file,
     "--line", str(line),
     "--snippet", snippet],
    env=dac_env,
    stdout=subprocess.PIPE, check=True,
    encoding="utf-8")
  return json.loads(p.stdout)

@click.group()
@click.option(
  "--base", "-b",
  type=click.STRING,
  default=PROJECTS_DIR)
@click.option(
  "--project", "-p",
  type=click.STRING)
@click.option(
  "--module",
  type=click.STRING)
@click.option(
  "--package",
  type=click.STRING)
@click.option(
  "--file",
  type=click.STRING)
@click.option(
  "--line",
  type=click.INT)
@click.option(
  "--snippet",
  type=click.STRING,
  default="")
@click.option(
  "--dist",
  type=click.INT,
  default=0)
@click.option(
  "--cache-dist",
  type=click.INT,
  default=3)
@click.option(
  "--go-version",
  type=click.STRING,
  default=GO_VERSION)
@click.option(
  "--convert-mode", "-c",
  type=click.STRING,
  default="atomic_blocks")
@click.pass_context
def cli(ctx, **opts):
  ctx.ensure_object(dict)
  if opts.get("module", None) is None:
    opts["module"] = opts["package"]
  ctx.obj.update(opts)

@cli.command()
@click.option(
  "--format", "-f",
  type=click.Choice(["json", "dot"]),
  default="json")
@click.pass_obj
def show(obj, format):
  cfg = get_cfg_json(**obj)

  if format == "json":
    print(json.dumps(cfg))
  elif format == "dot":
    inst = dict(cfg=cfg, usage=obj)
    g = dataset.raw_to_graphs(
      [inst], obj["convert_mode"], use_cache=False)[0]
    print(nx.nx_agraph.to_agraph(g).to_string())

@cli.command()
@click.option(
  "--model", "-m", required=True,
  type=click.STRING)
@click.option(
  "--limit-id", "-l",
  type=click.STRING,
  default="v127_d127_f127_p127")
@click.option(
  "--conformal-alpha", "-a",
  type=click.FloatRange(0.0, 1.0),
  default=0)
@click.option("--logits", is_flag=True, default=False)
@click.pass_obj
def predict(
  obj, model, limit_id, conformal_alpha=0.1, logits=False):
  with utils.cache_env(use_cache=False):
    convert_mode = obj["convert_mode"]
    cfg = get_cfg_json(**obj)
    inst = dict(cfg=cfg, usage=obj)
    graphs = dataset.raw_to_graphs(
      [inst], convert_mode)
    dir = f"{EXPORT_DIR}/{convert_mode}_{limit_id}/{model}"
    assert os.path.isdir(dir), "Requested model does not exist."
    with open(f"{EXPORT_DIR}/target_label_dims.json", "r") as f:
      labels1, labels2 = json.load(f)
      labels1_keys = list(labels1.keys())
      labels2_keys = list(labels2.keys())
    with open(f"{dir}/dims.json", "r") as f:
      dims = json.load(f)
    if conformal_alpha == 0.0:
      calib_config = dict(t1=1, t2=1)
    else:
      with open(f"{dir}/conformal_calibration_configs.yml", "r") as f:
        calib_configs = yaml.unsafe_load(f)
      assert conformal_alpha in calib_configs, f"Alpha must be from {calib_configs.keys()}."
      calib_config = calib_configs[conformal_alpha]
    in_enc = dims["in_enc"]
    encoder = dataset.dataset_encoders[in_enc]
    ds = encoder(graphs, dims)
    model = tf.keras.models.load_model(f"{dir}/model", custom_objects=dict(
      SparseMultiAccuracy=mm.SparseMultiAccuracy))
    l1_pred, l2_pred = model.predict(ds)
    l1_pred /= calib_config["t1"]
    l2_pred /= calib_config["t2"]
    if logits:
      prob1 = l1_pred[0]
      prob2 = l2_pred[0]
    else:
      prob1 = tf.nn.softmax(l1_pred, -1).numpy()[0]
      prob2 = tf.nn.softmax(l2_pred, -1).numpy()[0]
    l1_dict = fy.zipdict(labels1_keys, prob1)
    l2_dict = fy.zipdict(labels2_keys, prob2)

    if conformal_alpha == 0.0:
      print(json.dumps([l1_dict, l2_dict], cls=utils.NumpyEncoder))
    else:
      set1_idx = conf.adaptive_sets(l1_pred, calib_config["qhat1"])[0]
      set2_idx = conf.adaptive_sets(l2_pred, calib_config["qhat2"])[0]
      set1 = [labels1_keys[i] for i in set1_idx]
      set2 = [labels2_keys[i] for i in set2_idx]
      print(json.dumps(dict(
        probabilities=[l1_dict, l2_dict],
        conformal_sets=[set1, set2]), cls=utils.NumpyEncoder))


if __name__ == "__main__":
  cli(obj=dict())
