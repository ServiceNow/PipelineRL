# KUSHA: SET WANDB_ENTITY_NAME and wandb_workspace_root inside your .env
config="math_trial";
algo="grpo";
id=0;

while getopts c:a:i: flag
do
    case "${flag}" in
        c) config=${OPTARG};;
        a) algo=${OPTARG};;
        i) id=${OPTARG};;
    esac
done

echo $config $algo $id
source .env

export no_proxy=localhost,127.0.0.1,0.0.0.0,::1;
export PATH=$PATH:$HOME/redis-stable/src;

python -m pipelinerl.launch --config-name $config \
  finetune=$algo \
  output_dir="${SCRATCH}/pipeline-rl/results/${config}_${algo}_${id}" \
  wandb.wandb_entity_name=$WANDB_ENTITY_NAME \
  wandb.wandb_workspace_root=$WANDB_WORKSPACE_ROOT
