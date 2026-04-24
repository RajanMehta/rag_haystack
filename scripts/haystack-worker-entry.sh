#!/usr/bin/env bash

set -e

celery --app=haystack_api.tasks.app worker --concurrency=${CONCURRENCY:-2} --loglevel=${LOGLEVEL:-DEBUG}
