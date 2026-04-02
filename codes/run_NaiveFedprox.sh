nohup python -u main.py --model naive_fedprox --dataset fedpu_cifar10 --device_id 7 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --mu 1 --eval_pseudo > NaiveFedproxCIFARBs64Mu1.log 2>&1 &
 
nohup python -u main.py --model naive_fedprox --dataset fedpu_imagenette --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --mu 1 --eval_pseudo > NaiveFedproxImagenetteBs64Mu1.log 2>&1 &

nohup python -u main.py --model naive_fedprox --dataset fedpu_usps --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --mu 5 --eval_pseudo > NaiveFedproxUSPSBs64Mu5.log 2>&1 &

nohup python -u main.py --model naive_fedprox --dataset fedpu_letter --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --eval_pseudo > NaiveFedproxLetterBs64.log 2>&1 &
