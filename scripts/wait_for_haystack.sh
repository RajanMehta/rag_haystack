#!/usr/bin/env bash

# Waits for server dependencies to be healthy


function wait_for_dependency {
    name="$1"
    target="$2"

    wcnt=0
    inc=5
    limit=60
    while ! curl -s "${target}" > /dev/null; do
        wcnt=$((wcnt+inc))
        if [ "$wcnt" -gt "$limit" ]; then
            echo "Waited for ${limit} seconds. Quitting."
            exit 1
        fi
        echo "Waiting for ${name} server (${target}) to be ready. Waiting for" \
            "${inc} seconds (${wcnt}/${limit})."
        sleep "${inc}"
    done
    echo "${name} server (${target}) is ready after ${wcnt} seconds."
}

wait_for_dependency "${QDRANT_HOST}" "http://${QDRANT_HOST}:${QDRANT_HTTP_PORT}/metrics"
wait_for_dependency "haystack-api" "http://haystack-api:31415/health"