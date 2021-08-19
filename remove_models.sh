#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

EXPERIMENT_ID=${1:-"*"}
echo "Removing models and tensorboard logs for experiment $EXPERIMENT_ID."
read -p "Are you sure [yN]? " -n 1 -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
	echo
	echo "Aborted."
	exit 1
fi
echo
rm -rf ./mlruns/$EXPERIMENT_ID/*/artifacts/{models,tensorboard_logs}
echo "Done."
