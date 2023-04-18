#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=cnn_q49_30epoch_gpu

module purge

singularity exec \
	    --overlay /scratch/ct2840/env/my_pytorch38.ext3:ro \
	    /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif\
	    /bin/bash -c "source /ext3/env.sh; python cnn_runner.py"
