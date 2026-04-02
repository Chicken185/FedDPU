# CIFAR10
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05Warm30.log 2>&1 &

 nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 40 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05Warm40.log 2>&1 &

 nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05Warm50.log 2>&1 &

 nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05Warm60.log 2>&1 &

# Imagenette
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108Warm30.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 40 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108Warm40.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108Warm50.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108Warm60.log 2>&1 &

# USPS
nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 > FedpuUSPSWarm50Bs64Warm30.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 40 --weight_balance 0.5 > FedpuUSPSWarm50Bs64Warm40.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64Warm50.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuUSPSWarm50Bs64Warm60.log 2>&1 &


# Letter
nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 > FedpuLetterWarm50Bs96Warm30.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 40 --weight_balance 0.5 > FedpuLetterWarm50Bs96Warm40.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs96Warm50.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuLetterWarm50Bs96Warm60.log 2>&1 &