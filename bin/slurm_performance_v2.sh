#!/bin/bash
#SBATCH --job-name="perfpcv2"
#SBATCH -p rome16q #milanq #ipuq #milanq #armq #milanq #fpgaq #milanq # partition (queue)
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

echo "<performance study 4 PCNN.CORE>"

# activate the virtual environment
. /home/daniekru/codebase/myenvs/pcenv/bin/activate
echo "ecl1 activated"

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

# srun python3 analysis/study_performance.py --reps 5 --cores 64 --save --room "Square.v0"
srun python3 analysis/performance_comp_v2.py --reps 32 --cores 32 --save

echo "[finished]"

# --- PUSH
cd ~/lab/PCNN
git add .
git commit -m "+performance ex3"
git push

echo "[pushed]"

