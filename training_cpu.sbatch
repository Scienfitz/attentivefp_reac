#! /bin/bash -l

#SBATCH --cpus-per-task 22
#SBATCH --qos long
#SBATCH --job-name PyTorch_CPU_Training
#SBATCH --account cadd_services
#SBATCH --output /pstore/scratch/cadd/mlpp/pytorch_training.%A.out
#SBATCH --error /pstore/scratch/cadd/mlpp/pytorch_training.%A.err

ml --force purge
export PATH=/usr/bin:${PATH}

ml python/python3.7-2018.12
ml slurm
source /pstore/apps/Python/3.7.1-2018.12/etc/profile.d/conda.sh

conda activate dgllife-cpu

python /pstore/home/reutlinm/CODE/python/attentivefp/training.py "$@"
