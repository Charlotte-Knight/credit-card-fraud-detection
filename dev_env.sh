#!/usr/bin/env bash

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
  cat fastapi/requirements.txt synthetic_data/requirements.txt > $env_dir/requirements.txt
  micromamba env create -n $env_name -f $env_dir/requirements.txt
  micromamba activate $env_name
fi