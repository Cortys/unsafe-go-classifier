#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

if [ -z "$CONTAINER_NAME" ]; then
	CONTAINER_NAME="usgoc"
fi

docker exec -it $(docker ps -aqf "name=^$CONTAINER_NAME\$") mlflow ui -p 1234 -h 0.0.0.0
