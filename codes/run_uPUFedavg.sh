nohup python -u main.py --model upu_fedavg --dataset fedpu_cifar10 --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 > uPU_FedavgCIFARBs64.log 2>&1 &
 
nohup python -u main.py --model upu_fedavg --dataset fedpu_imagenette --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 8 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 > uPU_FedavgImagenetteBs8.log 2>&1 &

nohup python -u main.py --model upu_fedavg --dataset fedpu_usps --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 > uPU_FedavgUSPSBs64.log 2>&1 &

nohup python -u main.py --model upu_fedavg --dataset fedpu_letter --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 > uPU_FedavgLetterBS64.log 2>&1 &
###############################################################pseudo############################################################################################################################################################
nohup python -u main.py --model upu_fedavg --dataset fedpu_cifar10 --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --eval_pseudo > uPU_FedavgCIFARBs64.log 2>&1 &
 
nohup python -u main.py --model upu_fedavg --dataset fedpu_imagenette --device_id 6 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --eval_pseudo > uPU_FedavgImagenetteBs64.log 2>&1 &

nohup python -u main.py --model upu_fedavg --dataset fedpu_usps --device_id 7 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --eval_pseudo > uPU_FedavgUSPSBs64.log 2>&1 &

nohup python -u main.py --model upu_fedavg --dataset fedpu_letter --device_id 7 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --eval_pseudo > uPU_FedavgLetterBS64.log 2>&1 &
