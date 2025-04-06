#!/bin/bash
#SBATCH -J resnet                  # Job name
#SBATCH -G H100:1            # Number of nodes and cores per node required
#SBATCH -t 3:00:00                               # Duration of the job (8 hours)
#SBATCH --cpus-per-task=8
#SBATCH -o Report-%j.out                         # Combined output and error messages file
#SBATCH --output=slurm_out/Report-%A.out

# module load anaconda3/2022.05.0.1 
# conda activate flux
source .venv/bin/activate

echo "Running the following command:"
echo $@

srun $@
