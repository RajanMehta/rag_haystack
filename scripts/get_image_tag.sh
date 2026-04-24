#!/usr/bin/env bash

# Outputs the image tag used for tagging/pushing images

if [ "$CI_RELEASE_VERSION" ]; then
    echo $CI_RELEASE_VERSION # set in gitlab CI - this is for tagged releases
elif [ "$CI_COMMIT_SHA" ]; then
    echo $CI_COMMIT_SHA # set in gitlab CI - use commit SHA for normal pipelines
else
    echo 'latest' # for local builds, you shouldn't have to worry about tags.
fi
