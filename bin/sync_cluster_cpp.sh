#!/bin/bash

# --- compile cpp backend ---

# --- go to the right directory
cd ~/lab/PCNN/src/core

# --- activate the virtual environment
. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "ecl1 activated"

# --- compile
rm -rf build
mkdir build
./bin/make.sh

# --- run test
cd ..
pytest -v tests/test_model.py
cd ..

echo "[finished]"
