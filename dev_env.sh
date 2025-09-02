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
  micromamba env create -n $env_name -f ccfd-api/env.yaml -y
  micromamba install -n $env_name -f transaction-generator/env.yaml -y
  micromamba install -n $env_name ruff pre-commit -y
  micromamba clean --all --yes
  micromamba activate $env_name
fi