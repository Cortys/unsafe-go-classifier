#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

USER="-u $(id -u):$(id -g)"

if [ -z "$CONTAINER_NAME" ]; then
	CONTAINER_NAME="usgoc"
fi

CONTAINER_ID=$(docker ps -aqf "name=^$CONTAINER_NAME\$")
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(cat EXPERIMENT_NAME)}

echo "Pruning MLFLow db..."
docker exec -it $USER --workdir /app $CONTAINER_ID mlflow gc --backend-store-uri file:./mlruns
docker exec -it $USER -e EXPERIMENT_NAME=$EXPERIMENT_NAME --workdir /app $CONTAINER_ID mlflow_migrate/sqlite_migrate.sh
