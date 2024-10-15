#!/bin/bash


# ask the user for the type of agent
agent=$1

if [ -z "$agent" ]; then
  echo "Usage: $0 [agent]"
  echo "Agent [ppo, a2c, td3, ddpg, dqn]: "
  exit 1
fi

echo "Agent: $agent"
echo "% test %"
echo "--------"

# other settings
objective="target"
policy="forward"


# A2C
python3 src/simplerl/run.py --run "train" --duration 5_000 --new_pc --agent "$agent" --objective "$objective" --room "square" --test --load --dilation 10 --policy "$policy"

