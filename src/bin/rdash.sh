#!/bin/bash

echo "directory: $(pwd)"
# cd dashboard

# clean
echo "clearing cache..."
rm dashboard/cache/*

# run
echo "running dashboard..."
python3 -m http.server 8000

