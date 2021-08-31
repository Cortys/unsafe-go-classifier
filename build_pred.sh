#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

if [ -z "$CONTAINER_NAME" ]; then
	CONTAINER_NAME="usgoc/pred"
fi

docker build . -f Dockerfile.pred -t $CONTAINER_NAME $@
