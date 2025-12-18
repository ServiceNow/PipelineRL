algo="vapo";
# KUSHA: set your account in your .env
source .env

while getopts a: flag
do
    case "${flag}" in
        a) algo=${OPTARG};;
    esac
done

echo "Starting job: $algo"
NAME=""${algo}_8gpu""

sbatch <<EOT
#!/bin/bash

#SBATCH --account=$ACCOUNT
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=24
#SBATCH --nodes=2
#SBATCH --mem=480G
#SBATCH --output="$SCRATCH/pipeline-rl/logs/%j_$NAME.out"
#SBATCH --time=24:00:00
#SBATCH --job-name=$NAME

export HF_HOME="$SCRATCH/cache"
export NUM_GPUS=8

cd ~/PipelineRL

. tamia_activate.sh

source .env

bash scripts/run.sh -a $algo -c math_trial_8gpu
EOT