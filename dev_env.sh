env_dir=${PWD}/.dev_env
env_name=dev_env

if [ -d ".dev_env" ]; then
  export MAMBA_ROOT_PREFIX=$env_dir
  eval "$($env_dir/micromamba shell hook -s posix)"
  micromamba activate $env_name
else
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
  mv bin $env_dir
  export MAMBA_ROOT_PREFIX=$env_dir
  eval "$($env_dir/micromamba shell hook -s posix)"
  echo "python=3.13" > $env_dir/requirements.txt
  cat ccfd_api/requirements.txt transaction_generator/requirements.txt >> $env_dir/requirements.txt
  echo ruff >> $env_dir/requirements.txt
  echo pre-commit >> $env_dir/requirements.txt
  micromamba env create -n $env_name -f $env_dir/requirements.txt
  micromamba clean --all --yes
  micromamba activate $env_name
fi