#!/bin/bash

# clean
echo "clearing cache..."
rm -rf /dashboard/media/*

# run
echo "running dashboard..."
python3 -m http.server 8000

