#!/bin/bash
#SBATCH --job-name="pcg"
#SBATCH -p rome16q #milanq #ipuq #milanq #armq #milanq #fpgaq #milanq # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
##SBATCH --mem-per-cpu=1GB
#SBATCH --time=0-23:00
#SBATCH -o /home/daniekru/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/daniekru/slurm.column.%j.%N.err # STDERR


# --- SETUP

echo "<performance study 4 PCNN.CORE>"

# activate the virtual environment
. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "ecl1 activated"

# go to the right directory
cd ~/lab/PCNN/src/
echo "directory: $(ls)"
git checkout main
echo "[git 'main']"

# --- RUN

srun python3 analysis/study_gains.py --reps 4 --cores 32

echo "[finished]"

# --- PUSH
cd ~/lab/PCNN
git add .
git commit -m "+performance ex3"
git push

echo "[pushed]"

