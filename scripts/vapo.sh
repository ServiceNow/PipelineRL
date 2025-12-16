export no_proxy=localhost,127.0.0.1,0.0.0.0,::1;
python -m pipelinerl.launch --config-name math_vapo output_dir="${SCRATCH}/pipeline-rl/results/vapo" hydra.job.chdir=False