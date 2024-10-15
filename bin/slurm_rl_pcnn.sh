#!/bin/bash
#SBATCH --job-name="rlpc"
#SBATCH -p milanq #ipuq #milanq #armq #milanq #fpgaq #milanq # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
##SBATCH --mem-per-cpu=1GB
#SBATCH --time=0-18:00
#SBATCH -o /home/daniekru/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/daniekru/slurm.column.%j.%N.err # STDERR

# activate the virtual environment
. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "ecl1 activated"

# go to the right directory
cd ~/lab/PCNN/src
git checkout main
echo "[in main]"

# user selected agent
agent=$1

if [ -z "$agent" ]; then
  echo "Usage: $0 [agent]"
  echo "Agent [ppo, a2c, td3, ddpg, dqn]: "
  exit 1
fi

echo ">>> using agent '$agent'"

# other settings
objective="target"
policy="forward"

# Run the training
srun python3 simplerl/run.py --run "train" --epochs 1_000_000 --duration 6_000 --new_pc --agent "$agent" --objective "$objective" --room "square" --train --save --policy "$policy"

echo "[finished]"

# push
cd ~/lab/PCNN
git add .
git commit -m "training $agent"
git push

echo "[pushed]"

