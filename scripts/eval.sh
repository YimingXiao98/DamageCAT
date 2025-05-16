
gpus=0
img_size=512
data_name=DamageCATDataset
net_G=newUNetTrans
split=test
n_class=5
project_name=CROP_newUNetTrans_DamageCATDataset_b8_lr0.001_train_val_200_linear_ce_smoothen
checkpoint_name=best_ckpt.pt


python3 eval_cd.py --split ${split} --img_size ${img_size} --net_G ${net_G} --n_class ${n_class} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}


