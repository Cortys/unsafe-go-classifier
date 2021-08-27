import click

import usgoc.utils as utils

PROJECTS_DIR = "/projects"
EXPORT_DIR = f"{utils.PROJECT_ROOT}/exported_models"

@click.command()
@click.option(
  "--model", "-m",
  type=click.STRING)
@click.option(
  "--convert-mode", "-c",
  type=click.STRING,
  default="atomic_blocks")
@click.option(
  "--limit-id", "-l",
  type=click.STRING,
  default="v127_d127_f127_p127")
@click.option(
  "--base", "-b",
  type=click.STRING,
  default=PROJECTS_DIR)
@click.option(
  "--project", "-p",
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
  default="1.14.3")
def predict(
  model, convert_mode, limit_id,
  base, project, package, file, line, snippet,
  dist, cache_dist, go_version):
  print(
    model, convert_mode, limit_id, base, project, package, file, line, snippet,
    dist, cache_dist, go_version)


if __name__ == "__main__":
  predict()
