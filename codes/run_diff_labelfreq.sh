# CIFAR10
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.1 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05labelfreq01.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05labelfreq02.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.3 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05labelfreq03.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.4 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm50Bs16Weight05labelfreq04.log 2>&1 &

# Imagenette
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.1 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108labelfreq01.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108labelfreq02.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.3 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108labelfreq03.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.4 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs108labelfreq04.log 2>&1 &


# USPS
nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.1 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64labelfreq01.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64labelfreq02.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.3 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64labelfreq03.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.4 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64labelfreq04.log 2>&1 &

# Letter
nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 6 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.1 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs96labelfreq01.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 6 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs96labelfreq02.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 6 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.3 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs96labelfreq03.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 6 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.4 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs96labelfreq04.log 2>&1 &