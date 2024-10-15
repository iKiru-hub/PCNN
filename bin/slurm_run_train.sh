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

. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "ecl1 activated"

cd ~/lab/PCNN
git checkout main
echo "[in main]"

cd src/rl/smoothworld
echo "$(pwd)"

#srun python3 evo_main.py --verbose
srun python3 train.py  --EPOCHS 2_000_000 --log_int 2000 --T_TIMEOUT 2000 --GOAL_POS "0.3,0.3" --GOAL_RADIUS 0.3 --INIT_POS "0.7,0.7" --INIT_POS_RADIUS 0.2 --MAX_WALL_HITS 4 --save_frew 1_000_000 --flat one --save

echo "[finished]"


