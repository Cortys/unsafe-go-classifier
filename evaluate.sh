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

echo "Pruning MLFLow db..."
docker exec -it $USER $CUDA_ENV --workdir /app $CONTAINER_ID mlflow gc --backend-store-uri file:./mlruns

mlflow_migrate/sqlite_migrate.sh
