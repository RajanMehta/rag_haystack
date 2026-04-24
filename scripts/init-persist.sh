#!/bin/bash

set -e

DIRS=(
    haystack_api/file-upload
)

SUDO_UID=${DEFAULT_UID:-1000}
SUDO_GID=${DEFAULT_GID:-1000}

for d in "${DIRS[@]}"; do
    if [ ! -d "${d}" ]; then
        sudo mkdir -p "${d}"
    fi

    if [ "$(ls -lnd "${d}" | awk '{print $3 ":" $4}' )" != "${SUDO_UID:-"$(id -u)"}:${SUDO_GID:-"$(id -g)"}" ]; then
        sudo chown -R "${SUDO_UID:-"$(id -u)"}:${SUDO_GID:-"$(id -g)"}" "${d}"
    fi
done
