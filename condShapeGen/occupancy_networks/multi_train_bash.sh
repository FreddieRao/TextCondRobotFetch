#!/bin/bash
# compare with AE and stack ae
name[1]='CUDA_VISIBLE_DEVICES=0 bash occupancy_networks/train_bash.sh onet_table_vae256'
name[2]='CUDA_VISIBLE_DEVICES=1 bash occupancy_networks/train_bash.sh onet_table_vae128' 
name[3]='CUDA_VISIBLE_DEVICES=0 bash occupancy_networks/train_bash.sh onet_chair_vae256'
name[4]='CUDA_VISIBLE_DEVICES=1 bash occupancy_networks/train_bash.sh onet_chair_vae128' 
name[5]='CUDA_VISIBLE_DEVICES=0 bash occupancy_networks/train_bash.sh onet_table_z128' 
name[6]='CUDA_VISIBLE_DEVICES=1 bash occupancy_networks/train_bash.sh onet_chair_z128' 

Pfifo="/tmp/$$.fifo"
mkfifo $Pfifo
exec 6<>$Pfifo
rm -f $Pfifo

for i in $(seq 1 2)
do
	echo
done >&6

for lambdaq in $(seq 3 4)
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
