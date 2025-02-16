#!/bin/bash

# This script is used to build the project
# ---

# re-make build directory
rm -rf build
mkdir build
echo "> 'build' directory re-created"

# build the project
cd build
cmake ..
make
cd ..

echo "---"
echo "> build complete"


