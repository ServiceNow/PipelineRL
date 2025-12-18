# this documents roughly my process setting up the library on compute canada, please don't try to run the script but follow the process step-by-step

module load StdEnv/2023 rust/1.91.0
module load httpproxy
module load gcc
module load cuda/12.6
module load arrow/21.0.0 python/3.11 opencv/4.11
module load scipy-stack/2025a
module load python-build-bundle/2023b

python -m venv $SCRATCH/envs/prl
. $SCRATCH/envs/prl/bin/activate

pip install torch==2.6.0 torchvipision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install playwright-1.44.0-py3-none-linux_x86_64.whl
pip install peft==0.12.0
pip install -e . --no-build-isolation

pip install numpy==2.2.2+computecanada --force-reinstall
pip install platformdirs==3.10.0 --force-reinstall
pip install packaging==23.1 --force-reinstall
pip install pandas==2.2.3+computecanada --force-reinstall

# also tried changing this external thing but it didn't work
pip install "triton<3.3.0"
pip install datasets==2.21.0

# redis installation
cd $HOME
wget https://download.redis.io/redis-stable.tar.gz
tar -xzvf redis-stable.tar.gz
cd redis-stable
make