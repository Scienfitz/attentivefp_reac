#! /bin/bash -l

#SBATCH --cpus-per-task 6
#SBATCH --qos normal
#SBATCH --job-name PyTorch_Training
#SBATCH --partition gpu
#SBATCH --account cadd_projects
#SBATCH --gres gpu:v100:1
#SBATCH --output /pstore/scratch/cadd/mlpp/pytorch_training.%A.out
#SBATCH --error /pstore/scratch/cadd/mlpp/pytorch_training.%A.err

ml --force purge
export PATH=/usr/bin:${PATH}

ml python/python3.7-2018.12
ml slurm
source /pstore/apps/Python/3.7.1-2018.12/etc/profile.d/conda.sh

conda activate dgllife

python /pstore/home/reutlinm/CODE/python/attentivefp/training.py "$@"
