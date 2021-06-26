#!/usr/bin/env bash

USER="-u $(id -u):$(id -g)"

if [ "$1" == "root" ]; then
	USER="-u 0:0"
fi

if [ -z "$CONTAINER_NAME" ]; then
	CONTAINER_NAME="usgoc"
fi

docker exec -it $USER $(docker ps -aqf "name=^$CONTAINER_NAME\$") bash
