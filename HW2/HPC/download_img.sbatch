#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --mem=64GB
#SBATCH --output=download_full_img_%A.out
#SBATCH --error=download_full_img_%A.err
#SBATCH --job-name=test_download_img
#SBATCH --mail-type=END
#SBATCH --mail-user=ct2840@nyu.edu

module purge
module load python/intel/3.8.6
python3 batch_download_zips.py
