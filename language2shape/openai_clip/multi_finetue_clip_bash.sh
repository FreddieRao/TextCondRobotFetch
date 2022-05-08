#!/bin/bash
# compare with AE and stack ae
name[1]='CUDA_VISIBLE_DEVICES=0 python openai_clip/finetune_clip.py --cate chair --loss_name_list scontrast --loss_weights 1'
name[2]='CUDA_VISIBLE_DEVICES=1 python openai_clip/finetune_clip.py --cate chair --loss_name_list scontrast_cos --loss_weights 0.5_0.5'
name[3]='CUDA_VISIBLE_DEVICES=1 python openai_clip/finetune_clip.py --cate table --n_models MLPS --loss_name_list scontrast_cos_disdiv --loss_weights 0.9_0.09_0.01'
name[4]='CUDA_VISIBLE_DEVICES=3 python openai_clip/finetune_clip.py  --split_by_text  --loss_name_list scontrast_cos --loss_weights 0.5_0.5'
name[5]='CUDA_VISIBLE_DEVICES=4 python openai_clip/finetune_clip.py  --split_by_text  --loss_name_list scontrast_cos_disdiv --loss_weights 0.9_0.09_0.01'
name[6]='CUDA_VISIBLE_DEVICES=5 python openai_clip/finetune_clip.py  --split_by_text  --loss_name_list scontrast_mse_disdiv --loss_weights 0.5_0.25_0.25'
name[7]='CUDA_VISIBLE_DEVICES=6 python openai_clip/finetune_clip.py  --split_by_text  --loss_name_list scontrast_mse_disdiv --loss_weights 0.5_0.25_0.25'
name[8]='CUDA_VISIBLE_DEVICES=7 python openai_clip/finetune_clip.py  --split_by_text  --loss_name_list scontrast_mse_disdiv --loss_weights 0.5_0.25_0.25'
name[9]='CUDA_VISIBLE_DEVICES=0 python openai_clip/finetune_clip.py --orthogonal --norm LN --act_f hswish --n_epochs 48 --warmup_step 0'
name[10]='CUDA_VISIBLE_DEVICES=1 python openai_clip/finetune_clip.py --loss_name_list scontrast_mse --loss_weights 0.3_0.7'
name[11]='CUDA_VISIBLE_DEVICES=2 python openai_clip/finetune_clip.py --loss_name_list scontrast_mse --loss_weights 0.2_0.8'
name[12]='CUDA_VISIBLE_DEVICES=3 python openai_clip/finetune_clip.py --loss_name_list scontrast_mse --loss_weights 0.1_0.9'

Pfifo="/tmp/$$.fifo"
mkfifo $Pfifo
exec 6<>$Pfifo
rm -f $Pfifo

for i in $(seq 1 1)
do
	echo
done >&6

for lambdaq in $(seq 3 3)
	do
	read -u 6
	{
	eval ${name[${lambdaq}]}
	sleep 1
	echo >&6
	}&
done

wait
exec 6 >
