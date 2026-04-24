#!/usr/bin/env bash

set -e

uvicorn --app-dir=haystack_api/ application:app --host 0.0.0.0 --port ${PORT:-31415}
