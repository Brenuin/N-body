#!/bin/bash
#SBATCH -J NBodySimulation       # Job name
#SBATCH -A cs475-575             # Account name
#SBATCH -p classgputest          # Partition name
#SBATCH --constraint=v100        # GPU constraint
#SBATCH --gres=gpu:1             # Number of GPUs
#SBATCH -o nbody.out             # Standard output file
#SBATCH -e nbody.err             # Standard error file
#SBATCH --mail-type=BEGIN,END,FAIL # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=santossk@oregonstate.edu # Your email address

# Load the required module
module load cuda/11.7

# Compile the CUDA program
/usr/local/apps/cuda/11.7/bin/nvcc -DTIME=1000 -DNUMPLANETS=1050 -DNUMSTARS=500 -DNUMSMALLBLACKHOLES=20 -DNUMBLACKHOLES=10 -DGALAXYR=150.0e9 -o nbody nbody.cu
./nbody


# Run the program
./nbody
