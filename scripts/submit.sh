run_script="vapo";

while getopts r: flag
do
    case "${flag}" in
        r) run_script=${OPTARG};;
    esac
done

echo "Starting job: $run_script"
NAME=""${run_script}""

sbatch <<EOT
#!/bin/bash

#SBATCH --account=aip-siamakx
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=24
#SBATCH --nodes=1
#SBATCH --mem=480G
#SBATCH --output="$SCRATCH/pipeline-rl/logs/%j_$NAME.out"
#SBATCH --time=24:00:00
#SBATCH --job-name=$NAME

export HF_HOME="$SCRATCH/cache"
export NUM_GPUS=4

cd ~/PipelineRL

. tamia_activate.sh

source .env

bash scripts/${run_script}.sh
EOT
