#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=ct2840@nyu.edu
#SBATCH --job-name=resnet_hw2_gpu
#SBATCH --output=slurm_%j_resnet_gpu.out

module purge

singularity exec --nv\
	    --overlay /scratch/ct2840/env/my_pytorch38.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python resnet_runner.py"
