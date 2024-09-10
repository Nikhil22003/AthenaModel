#!/usr/bin/env bash

set -x

docker rm -f cog-wav2lip

docker run -d --restart always \
  --name cog-wav2lip \
  -v $PWD/checkpoints:/src/checkpoints \
  -p 5001:5000 \
  --gpus all \
  r8.im/devxpy/cog-wav2lip

docker logs -f cog-wav2lip
