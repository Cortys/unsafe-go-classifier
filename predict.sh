#!/usr/bin/env bash

cd "${BASH_SOURCE%/*}" || exit

pd=$(realpath ${PROJECTS_DIR:-"./unsafe_go_tools/projects"})
docker run --rm \
	-v go_mod:/root/go/pkg/mod -v $pd:/projects \
	-v $(pwd):/app \
	usgoc/pred:latest $@
