#!/bin/bash


duration=90
seed=187
main_idx=1


# Run your program (replace with actual program)
python3 main.py --duration $duration --seed $seed --main $main_idx --animate --verbose &

# Get the PID of the program
pid=$!

echo "PID: $pid"

sleep 1

# Limit CPU usage to 50% (you can adjust the percentage as needed)
cpulimit -p $pid -l 50

