#!/bin/bash
#SBATCH --job-name="modpc"
#SBATCH -p rome16q #milanq #ipuq #milanq #armq #fpgaq partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
##SBATCH --mem-per-cpu=1GB
#SBATCH --time=0-23:00
#SBATCH -o /home/daniekru/slurm.column.%j.%N.out # STDOUT
#SBATCH -e /home/daniekru/slurm.column.%j.%N.err # STDERR


# --- SETUP
module purge
module load slurm/21.08.8

echo "<modulation study>"

# activate the virtual environment
. /home/daniekru/codebase/myenvs/ecl1/bin/activate
echo "venv activated"

# go to the right directory
cd ~/lab/PCNN/src/
echo "directory: $(ls)"
git checkout main
echo "[git 'main']"

# --- RUN
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK}

srun python3 analysis/study_mod.py --reps 2 --cores 32 --save

echo "[finished]"

# --- PUSH
cd ~/lab/PCNN
git add .
git commit -m "+mod"
git push

echo "[pushed]"

