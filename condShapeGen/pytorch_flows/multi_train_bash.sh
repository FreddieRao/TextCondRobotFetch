#!/bin/bash
# compare with AE and stack ae
name[1]='CUDA_VISIBLE_DEVICES=0 bash pytorch_flows/train_bash.sh 164642751674410 table'
name[2]='CUDA_VISIBLE_DEVICES=1 bash pytorch_flows/train_bash.sh 164642751671820 table' 
name[3]='CUDA_VISIBLE_DEVICES=0 bash pytorch_flows/train_bash.sh 164653751614492 table'
name[4]='CUDA_VISIBLE_DEVICES=0 bash pytorch_flows/train_bash.sh 0001 table'
name[5]='CUDA_VISIBLE_DEVICES=0 bash pytorch_flows/train_bash.sh 0000 chair'
name[6]='CUDA_VISIBLE_DEVICES=1 bash pytorch_flows/train_bash.sh 164659151491333 chair'
name[7]='CUDA_VISIBLE_DEVICES=0 bash pytorch_flows/train_bash.sh 164644541495829 chair'
name[8]='CUDA_VISIBLE_DEVICES=1 bash pytorch_flows/train_bash.sh 164644541495830 chair'

Pfifo="/tmp/$$.fifo"
mkfifo $Pfifo
exec 6<>$Pfifo
rm -f $Pfifo

for i in $(seq 1 1)
do
	echo
done >&6

for lambdaq in $(seq 4 4)
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
