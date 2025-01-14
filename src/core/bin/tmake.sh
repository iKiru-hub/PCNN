#!/bin/bash

# This script is used to compile and run
# the file `src/test.cpp`
# ---


# build the project
cd build
cmake ..
make

echo "------"
echo "Compilation successful. Running the program..."
echo "------"
echo " "

./pcnn_test

echo " "
echo "------"
echo "."

