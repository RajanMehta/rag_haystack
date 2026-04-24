#!/usr/bin/env bash

set -e

/qdrant/entrypoint.sh "$@" 2>&1 | tee /var/log/service/qdrant.log
