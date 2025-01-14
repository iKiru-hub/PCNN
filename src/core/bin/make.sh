#!/bin/bash

# This script is used to build the project
# ---

# trg_path="/Users/daniekru/Research/lab/PCNN/src/libs"

# build the project
cd build
cmake ..
make
cd ..

echo "---"
echo "> build complete"

# copy the .so file to the target path
# cp build/*.so $trg_path

# echo "> copy complete"

