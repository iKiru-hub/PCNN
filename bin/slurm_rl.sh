#!/bin/bash
#SBATCH --job-name="rlpc"
#SBATCH -p rome16q #armq #milanq #fpgaq #milanq # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
##SBATCH --mem-per-cpu=2GB
#SBATCH --time=0-23:00
#SBATCH -o /home/daniekru/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/daniekru/slurm.column.%j.%N.err # STDERR


# --- SETUP

echo "<RL 4 PCNN.CORE>"

# activate the virtual environment
. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "ecl1 activated"

# go to the right directory
cd ~/lab/PCNN/src/
echo "directory: $(ls)"
git checkout main
echo "[git 'main']"

# --- RUN

srun python3 main_rl.py --duration 10_000 --room "Flat.1000x" --interval 5 --episodes 500

echo "[finished]"

# --- PUSH
cd ~/lab/PCNN
git add .
git commit -m "+rl ex3"
git push

echo "[pushed]"

