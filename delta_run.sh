#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16   # <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --time=00:45:00
#SBATCH --account=bbtb-delta-gpu
#SBATCH --job-name=OPRO-T2I
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=4
#SBATCH --gpu-bind=verbose,per_task:1
###SBATCH --gpu-bind=none   # <- or closest



# Do not need these module stuff, using conda, in delta:
# conda deactivate
# conda deactivate  # goes back to no base anymore
# module purge
# module reset
# module load anaconda3_gpu
# conda activate nlp_env
# AND then just run this script.

# module purge # drop modules and explicitly load the ones needed
# module load anaconda3_gpu
# module list # job documentation and metadata

# # Ensure Conda is initialized
# source ~/.bashrc
# source /u/zhenans2/.conda/envs/nlp_env/bin/activate

echo "Job is starting on $(hostname)"
echo "Python binary: $(which python)"
echo "Python version: $(python --version)"



# source ~/.bashrc
# conda activate nlp_env
# # source activate nlp_env

# echo "job is starting on `hostname`"
# which python

export SLURM_MPI_TYPE=pmi2

PRT_DIR="/scratch/bbtb/zhenans2/NLP_opro_t2i"

srun python /scratch/bbtb/zhenans2/NLP_opro_t2i/opro-T2I/opro/optimization/optimize_instructions_T2I.py \
       --optimizer="gpt-4o-mini" \
       --scorer="relevance" \
       --dataset="diffusionDB" \
       --save-dir='.' \
       --param-aggregate-scores=True \
       --param-subset-size 20 \
       --param-num-search-steps 20 \
       --param-num-gen-per-search 3 \
       --openai_api_key="<>" \
       --openai_api_base="<>"

exit
