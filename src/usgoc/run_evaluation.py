import click

import usgoc.evaluation.datasets as ed
import usgoc.evaluation.models as em
import usgoc.evaluation.evaluate as ee

@click.command()
@click.option(
  "--model", "-m",
  type=click.Choice(em.models.keys()),
  default=em.evaluate_models,
  multiple=True)
@click.option(
  "--limit-id", "-l",
  type=click.Choice(ed.limit_ids),
  default=ed.evaluate_limit_ids,
  multiple=True)
@click.option(
  "--fold", "-f",
  type=click.IntRange(0, ee.FOLDS_MAX),
  default=None)
@click.option(
  "--repeat", "-r",
  type=click.IntRange(0, ee.REPEATS_MAX),
  default=None)
@click.option("--override/--no-override", default=False)
@click.option("--dry/--no-dry", default=False)
@click.option("--suffix", type=click.STRING, default="")
@click.option("--yes", "-y", is_flag=True, default=False)
def evaluate(
  model, limit_id, fold=None, repeat=None,
  override=False, dry=False,
  suffix="", yes=False):
  print("Starting evaluation.")
  print(f"Will use the following {len(model)} models:")
  for m in model:
    print(f"- {m}")
  print(f"Will use the following {len(limit_id)} limit_ids:")
  for lid in limit_id:
    print(f"- {lid}")
  print("Other options:")
  print(f"- Fold: {str(fold)}")
  print(f"- Repeat: {str(repeat)}")
  print(f"- Override: {str(override)}")
  print(f"- Dry run: {str(dry)}")
  print(f"- Experiment suffix: \"{suffix}\"")

  if not yes:
    click.confirm("Continue?", default=True, abort=True)

  print("----------------------------------------------------------\n")

  for m in model:
    ee.evaluate(
      m, limit_ids=limit_id,
      fold=fold,
      repeat=repeat,
      override=override, dry=dry,
      experiment_suffix=suffix)
    print("\n----------------------------------------------------------\n")

  print("Evaluation completed.")


if __name__ == "__main__":
  evaluate()
