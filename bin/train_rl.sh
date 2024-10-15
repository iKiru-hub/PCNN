#!/bin/bash


# ask the user for the type of agent
agent=$1

if [ -z "$agent" ]; then
  echo "Usage: $0 [agent]"
  echo "Agent [ppo, a2c, td3, ddpg, dqn]: "
  exit 1
fi

echo "Agent: $agent"
echo "% train %"
echo "---------"

# other settings
objective="target"
policy="forward"

#
python3 src/simplerl/run.py --run "train" --epochs 20_000 --duration 50_000 --dilation 20 --new_pc --agent "$agent" --objective "$objective" --room "square" --train --save --policy "$policy"
