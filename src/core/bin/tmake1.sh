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
echo "c++ test:"

./pcnn_test

echo " "
echo "------"
echo "python test:"

cd ..
ecl1
pytest -v testlib/test1.py

