#!/bin/bash

# Function to perform GET requests
perform_requests() {

    echo ""
    echo "CREATE Collection"
    curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "institution_id": "test",
        "collection_name": "collection_1"
    }' \
    http://haystack-api:31415/collection/create

    echo ""
    echo "GET Collection"
    curl -X GET http://haystack-api:31415/collection/test/collection_1

    echo ""
    echo "DELETE Collection"
    curl -X DELETE http://haystack-api:31415/collection/test/collection_1

}

# Call the function to perform all requests
perform_requests
