export PYTHONPATH=$(pwd)
module load StdEnv/2023 rust/1.91.0
module load httpproxy
module load gcc
module load cuda/12.6
module load arrow/21.0.0 python/3.11 opencv/4.11
module load scipy-stack/2025a
module load python-build-bundle/2023b

. $SCRATCH/envs/prl/bin/activate
