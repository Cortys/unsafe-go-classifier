#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

if [ -z "$CONTAINER_NAME" ]; then
	CONTAINER_NAME="usgoc/prod"
fi

docker build . -f Dockerfile.prod -t $CONTAINER_NAME
