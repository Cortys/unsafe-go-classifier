#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

USER="-u $(id -u):$(id -g)"
CUDA_ENV=""

if [ ! -z "$CUDA_VISIBLE_DEVICES" ]; then
	CUDA_ENV="-e CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

if [ -z "$CONTAINER_NAME" ]; then
	CONTAINER_NAME="usgoc"
fi

CONTAINER_ID=$(docker ps -aqf "name=^$CONTAINER_NAME\$")

echo "Starting evaluation script..."
docker exec -it $USER $CUDA_ENV --workdir /app/src $CONTAINER_ID python3 ./usgoc/run_evaluation.py $@ #\
	# | grep --line-buffered -vE \
	# "BaseCollectiveExecutor::StartAbort|IteratorGetNext|Shape/|Shape_[0-9]+/"

succ=$?

if [ ! -z "$NO_EXPORT" ]; then
	echo "No MLFLow DB export."
elif [ $succ -eq 0 ]; then
	echo "Pruning MLFLow db..."
	EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(cat EXPERIMENT_NAME)}
	docker exec -it $USER --workdir /app $CONTAINER_ID mlflow gc --backend-store-uri file:./mlruns
	docker exec -it $USER -e EXPERIMENT_NAME=$EXPERIMENT_NAME -e ONLY_LAST_METRIC=1 --workdir /app $CONTAINER_ID mlflow_migrate/sqlite_migrate.sh
else
	echo "Evaluation failed."
	exit 1
fi
