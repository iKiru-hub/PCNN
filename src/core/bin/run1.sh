#!/bin/bash

clear

# run web server
echo "running dashboard..."
python3 -m http.server 8000 &

# run simulation
echo "running simulation..."
python3 run_core.py --main "main" --N 100 --duration 40000 --seed -2 &


