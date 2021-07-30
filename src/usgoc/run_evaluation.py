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
  "--convert-mode", "-c",
  type=click.Choice(ed.convert_modes),
  default=ed.evaluate_convert_modes,
  multiple=True)
@click.option(
  "--limit-id", "-l",
  type=click.Choice(ed.limit_ids),
  default=ed.evaluate_limit_ids,
  multiple=True)
@click.option(
  "--tuner-convert-mode",
  type=click.Choice(ed.convert_modes),
  default=None)
@click.option(
  "--tuner-limit-id",
  type=click.Choice(ed.limit_ids),
  default=None)
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
  model, convert_mode, limit_id, fold=None, repeat=None,
  override=False, dry=False,
  tuner_convert_mode=None, tuner_limit_id=None,
  suffix="", yes=False):
  print("Starting evaluation.")
  print(f"Will use the following {len(model)} models:")
  for m in model:
    print(f"- {m}")
  print(f"Will use the following {len(convert_mode)} convert modes:")
  for m in convert_mode:
    print(f"- {m}")
  print(f"Will use the following {len(limit_id)} limit_ids:")
  for lid in limit_id:
    print(f"- {lid}")
  print("Other options:")
  print(f"- Fold: {str(fold)}")
  print(f"- Repeat: {str(repeat)}")
  print(f"- Tuner convert mode: {str(tuner_convert_mode)}")
  print(f"- Tuner limit_id: {str(tuner_limit_id)}")
  print(f"- Override: {str(override)}")
  print(f"- Dry run: {str(dry)}")
  print(f"- Experiment suffix: \"{suffix}\"")

  if not yes:
    click.confirm("Continue?", default=True, abort=True)

  print("----------------------------------------------------------\n")

  for m in model:
    ee.evaluate(
      m,
      convert_modes=convert_mode,
      limit_ids=limit_id,
      fold=fold,
      repeat=repeat,
      override=override, dry=dry,
      experiment_suffix=suffix,
      tuner_convert_mode=tuner_convert_mode,
      tuner_limit_id=tuner_limit_id,
      return_models=False)
    print("\n----------------------------------------------------------\n")

  print("Evaluation completed.")


if __name__ == "__main__":
  evaluate()
