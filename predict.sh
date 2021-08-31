#!/usr/bin/env bash

pd=$(realpath ${PROJECTS_DIR:-"${BASH_SOURCE%/*}/unsafe_go_tools/projects"})

if [ "$DEV" == "1" ]; then
	ARGS="-v $(pwd):/app"
else
	ARGS="-e TF_CPP_MIN_LOG_LEVEL=3"
fi

docker run --rm $ARGS \
	-v go_mod:/root/go/pkg/mod -v go_cache:/root/.cache/go-build -v $pd:/projects \
	usgoc/pred:latest $@
