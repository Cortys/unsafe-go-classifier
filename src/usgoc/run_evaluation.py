import click

import usgoc.evaluation.datasets as ed
import usgoc.evaluation.utils as eu
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
  multiple=True, show_choices=False, hidden=True)
@click.option(
  "--tuner-convert-mode",
  type=click.Choice(ed.convert_modes),
  default=None)
@click.option(
  "--tuner-limit-id",
  type=click.Choice(ed.limit_ids),
  default=None, show_choices=False, hidden=True)
@click.option(
  "--fold", "-f",
  type=click.IntRange(0, eu.FOLDS_MAX),
  default=None)
@click.option(
  "--repeat", "-r",
  type=click.IntRange(0, eu.REPEATS_MAX),
  default=None)
@click.option("--override/--no-override", default=False)
@click.option("--override-conformal/--no-override-conformal", default=False)
@click.option("--dry/--no-dry", default=False)
@click.option("--suffix", type=click.STRING, default="")
@click.option("--yes", "-y", is_flag=True, default=False)
@click.option("--export", "-e", is_flag=True, default=False)
@click.option("--confusion-matrices", is_flag=True, default=False)
@click.option("--nop", is_flag=True, default=False)
def evaluate(
  model, convert_mode, limit_id, fold=None, repeat=None,
  override=False, override_conformal=False, dry=False,
  tuner_convert_mode=None, tuner_limit_id=None,
  suffix="", yes=False, export=False, confusion_matrices=False, nop=False):
  if nop:
    print("NO OP: No evaluation done.")
    return

  if export:
    print("Starting export of winning models.")
    f = ee.export_best
  elif confusion_matrices:
    print("Starting export of confusion matrices.")
    f = ee.export_confusion_matrices
  else:
    print("Starting evaluation.")
    f = ee.evaluate
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
  print(f"- Override conformal: {str(override_conformal)}")
  print(f"- Dry run: {str(dry)}")
  print(f"- Experiment suffix: \"{suffix}\"")

  if not yes:
    click.confirm("Continue?", default=True, abort=True)

  print("----------------------------------------------------------\n")

  for m in model:
    f(
      m,
      convert_modes=convert_mode,
      limit_ids=limit_id,
      fold=fold,
      repeat=repeat,
      override=override, override_conformal=override_conformal,
      dry=dry,
      experiment_suffix=suffix,
      tuner_convert_mode=tuner_convert_mode,
      tuner_limit_id=tuner_limit_id)
    print("\n----------------------------------------------------------\n")

  if export:
    print("Exported models.")
  elif confusion_matrices:
    print("Exported confusion matrices.")
  else:
    print("Evaluation completed.")


if __name__ == "__main__":
  evaluate()
