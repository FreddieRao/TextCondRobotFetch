#!/bin/bash
# compare with AE and stack ae
name[1]='CUDA_VISIBLE_DEVICES=0 bash openai_clip/inference_shapenet_bash.sh 1645468753585 1645468753585_1645485794876 finetune_clip'
name[2]='CUDA_VISIBLE_DEVICES=1 bash openai_clip/inference_shapenet_bash.sh 164644541494783 164644541494783_1646445955565 finetune_clip'  
name[3]='CUDA_VISIBLE_DEVICES=0 bash openai_clip/inference_shapenet_bash.sh 164644541495829 164644541495829_1646445955587 finetune_clip'
name[4]='CUDA_VISIBLE_DEVICES=1 bash openai_clip/inference_shapenet_bash.sh 164644541495830 164644541495830_1646446234652 finetune_clip'
name[5]='CUDA_VISIBLE_DEVICES=0 bash openai_clip/inference_shapenet_bash.sh 0001 0001_1646613795015 no_finetune_clip'
name[6]='CUDA_VISIBLE_DEVICES=1 bash openai_clip/inference_shapenet_bash.sh 0001 0001_1645899608857 no_finetune_clip'
name[7]='CUDA_VISIBLE_DEVICES=0 bash openai_clip/inference_shapenet_bash.sh 164642751671820 164642751671820_1646431196168 finetune_clip '
name[8]='CUDA_VISIBLE_DEVICES=0 bash openai_clip/inference_shapenet_bash.sh 164659151491333 0001_1646617089644 finetune_clip'
name[9]='CUDA_VISIBLE_DEVICES=0 bash openai_clip/inference_shapenet_bash.sh 164653355155470 164653355155470_1646535291684 finetune_clip _t15_res128'
name[10]='CUDA_VISIBLE_DEVICES=1 bash openai_clip/inference_shapenet_bash.sh 164653355155470 164653355155470_1646535291684 finetune_clip _t15_res128_cut'
name[11]='CUDA_VISIBLE_DEVICES=0 bash openai_clip/inference_shapenet_bash.sh 164654299387510 164654299387510_1646544388259 finetune_clip _t05_res128'
name[12]='CUDA_VISIBLE_DEVICES=1 bash openai_clip/inference_shapenet_bash.sh 164654299387510 164654299387510_1646544388259 finetune_clip _t05_res128_cut'

Pfifo="/tmp/$$.fifo"
mkfifo $Pfifo
exec 6<>$Pfifo
rm -f $Pfifo

for i in $(seq 1 1)
do
	echo
done >&6

for lambdaq in $(seq 1 1)
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
