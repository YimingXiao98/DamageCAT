#!/usr/bin/env bash

gpus=0
checkpoint_root=checkpoints 
data_name=DamageCATDataset
dataset=DamageCATDataset
loss=ce_dice
n_class=5
lr=0.001
lr_policy=linear

img_size=512
batch_size=8

max_epochs=300
net_G=newUNetTrans

random_seed=2026

split=train  
split_val=val  
project_name=seed${random_seed}_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_300_${lr_policy}_ce_dice_smoothen

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr} --dataset ${dataset} --loss ${loss} --n_class ${n_class} --random_seed ${random_seed}