#!/usr/bin/env bash

set -ex

docker run --rm \
  --name cog-wav2lip \
  -v $PWD/checkpoints:/src/checkpoints \
  -p 5001:5000 \
  --gpus all \
  r8.im/devxpy/cog-wav2lip
