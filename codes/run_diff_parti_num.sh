# CIFAR10sh
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 25 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05Parti25.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05Parti50.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 75 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05Parti75.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 100 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05Parti100.log 2>&1 &
# Imagenette
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 25 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108Parti25.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108Parti50.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 75 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108Parti75.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 100 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108Parti100.log 2>&1 &

# USPS
nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 25 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64Parti25.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64Parti50.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 75 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64Parti75.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 100 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64Parti100.log 2>&1 &

# Letter
nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 25 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs96Parti25.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs96Parti50.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 75 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs96Parti75.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 100 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs96Parti100.log 2>&1 &