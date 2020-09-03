#! /bin/bash -l

#SBATCH --cpus-per-task 2
#SBATCH --qos short
#SBATCH --job-name PyTorch_Inference
#SBATCH --account cadd_projects
#SBATCH --output /pstore/scratch/cadd/mlpp/pytorch_predict.%A.out
#SBATCH --error /pstore/scratch/cadd/mlpp/pytorch_predict.%A.err

ml --force purge
export PATH=/usr/bin:${PATH}

ml python/python3.7-2018.12
ml slurm
source /pstore/apps/Python/3.7.1-2018.12/etc/profile.d/conda.sh

conda activate dgllife-cpu

python /pstore/home/reutlinm/CODE/python/attentivefp/predict.py "$@"
