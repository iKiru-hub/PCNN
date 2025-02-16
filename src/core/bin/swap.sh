#!/bin/bash

# set the correct cmake file in place

# ask user for the cmake file to use: "linux" or "mac"
echo "CMake to use: linux [0] or mac [1]? "
read ans

# check the user input
if [ $ans -eq 0 ]; then
    echo "Using linux cmake file"
    cp tmp/cmake_for_linux.txt CMakeLists.txt
elif [ $ans -eq 1 ]; then
    echo "Using mac cmake file"
    cp tmp/cmake_for_macos.txt CMakeLists.txt
else
    echo "Invalid input"
fi

echo "[done]"
