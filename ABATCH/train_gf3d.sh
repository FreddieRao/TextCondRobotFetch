#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -N 1
#SBATCH -n 16 --mem=128g -p "3090-gcondo" --gres=gpu:2
#SBATCH -t 48:00:00
#SBATCH -o ABATCH/out/self_position_embedding_none.out

cd Group-Free-3D
bash train_bash.sh

