# KUSHA: SET WANDB_ENTITY_NAME and wandb_workspace_root inside your .env
config="math_trial";
algo="grpo";

while getopts c:a: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
        a) algo=${OPTARG};;
    esac
done

echo $config $algo
source .env

export no_proxy=localhost,127.0.0.1,0.0.0.0,::1;
python -m pipelinerl.launch --config-name $config \
  finetune=$algo \
  output_dir="${SCRATCH}/pipeline-rl/results/math_${algo}" \
  wandb.wandb_entity_name=$WANDB_ENTITY_NAME \
  wandb.wandb_workspace_root=$WANDB_WORKSPACE_ROOT
