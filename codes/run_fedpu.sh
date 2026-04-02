# 对public_size的设定
# CIFAR-10	    50,000	500	 500	每类抽取 50 张
# ImageNette	9,469	~95	 100	每类抽取 10 张
# USPS	        7,291	~73	 100	每类抽取 10 张
# Letter	    16,000	160	 260    每类抽取 10 行 (共26类)
# 对label_frequency的设定：label_freq=0.2
# 对pos_class_list的设定：
# CIFAR-10 0 1 2 8 9
# Imagenette 0 1 2 8 9
# USPS 4 7 9 5 8
# Letter 1 21 11 17 8 14 22 18 9 10 2 7 25
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 128 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuCIFARWarm50Bs128Weight05.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuCIFARWarm50.log 2>&1 &
 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 240 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 40 --weight_balance 0.5 > FedpuImagenetteWarm40Bs240.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 > FedpuUSPSWarm30.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuUSPSWarm60Bs16.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 20 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuLetterWarm50Bs20.log 2>&1 &
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
##prior only
 nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --consistency_mode prior_only > FedpuCIFARPriorWarm10Bs64.log 2>&1 &
 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 7 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 --consistency_mode prior_only > FedpuImagenettePriorWarm60Bs16.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode prior_only > Fedpu_PriorUSPSWarm10.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --consistency_mode prior_only > FedpuLetterPriorWarm10Bs96.log 2>&1 &
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# feature only
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode feature_only > FedpuCIFARFeatureWarm30.log 2>&1 &
 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode feature_only > FedpuImagenetteFeatureWarm30.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode feature_only > FedpuUSPSFeatureWarm30.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 3 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.5 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode feature_only > FedpuLetterFeatureWarm30.log 2>&1 &
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# beat=0.25
#---------------------------------------------------------------------------------------------------------------
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 7 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm60Bs64Weight05.log 2>&1 &

 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 7 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 72 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs72.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 > FedpuUSPSWarm50Bs64.log 2>&1 &


nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 5 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 48 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 60 --weight_balance 0.9 > FedpuLetterWarm60Bs48Weight09.log 2>&1 &
###
##prior only
 nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 --consistency_mode prior_only > FedpuCIFARPriorWarm50Bs64.log 2>&1 &
 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode prior_only > FedpuImagenettePriorWarm30Bs64.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode prior_only > Fedpu_PriorUSPSWarm10Bs64.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --consistency_mode prior_only > FedpuLetterPriorWarm10Bs64.log 2>&1 &
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# feature only
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 --consistency_mode feature_only > FedpuCIFARFeatureWarm50Bs64.log 2>&1 &
 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode feature_only > FedpuImagenetteFeatureWarm30Bs64.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode feature_only > FedpuUSPSFeatureWarm30Bs64.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.25 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode feature_only > FedpuLetterFeatureWarm30Bs64.log 2>&1 &

————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# beat=0.75
#---------------------------------------------------------------------------------------------------------------
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuCIFARWarm60Bs64Weight05.log 2>&1 &

 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 7 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 112 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuImagenetteWarm60Bs112.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 120 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 > FedpuUSPSWarm60Bs120.log 2>&1 &


nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 60 --weight_balance 0.9 > FedpuLetterWarm60Bs108Weight09.log 2>&1 &
###
##prior only
 nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 7 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode prior_only > FedpuCIFARPriorWarm30Bs64.log 2>&1 &
 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 20 --weight_balance 0.5 --consistency_mode prior_only > FedpuImagenettePriorWarm20Bs64.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 32 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode prior_only > Fedpu_PriorUSPSWarm10Bs32.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 4 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --consistency_mode prior_only > FedpuLetterPriorWarm10Bs96.log 2>&1 &
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# feature only
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --consistency_mode feature_only > FedpuCIFARFeatureWarm10Bs64.log 2>&1 &
 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 20 --weight_balance 0.5 --consistency_mode feature_only > FedpuImagenetteFeatureWarm20Bs64.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --consistency_mode feature_only > FedpuUSPSFeatureWarm10Bs64.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --consistency_mode feature_only > FedpuLetterFeatureWarm10Bs64.log 2>&1 &


#########################################################Pseudo#####################################################
# FedPU
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 --eval_pseudo  > FedpuCIFARWarm50Bs16Weight05Online02.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 108 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 --eval_pseudo  > FedpuImagenetteWarm60Bs108Online02.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 --eval_pseudo  > FedpuUSPSWarm50Bs64Online02.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 0 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 50 --weight_balance 0.5 --eval_pseudo  > FedpuLetterWarm50Bs96Online02.log 2>&1 &

#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
##prior only
 nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --consistency_mode prior_only --eval_pseudo > FedpuCIFARPriorWarm10Bs64.log 2>&1 &
 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 16 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 60 --weight_balance 0.5 --consistency_mode prior_only --eval_pseudo > FedpuImagenettePriorWarm60Bs16.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode prior_only --eval_pseudo > Fedpu_PriorUSPSWarm10.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 1 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 96 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 10 --weight_balance 0.5 --consistency_mode prior_only --eval_pseudo > FedpuLetterPriorWarm10Bs96.log 2>&1 &
#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
# feature only
nohup python -u main.py --model fedpu --dataset fedpu_cifar10 --device_id 6 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 500 --label_freq 0.2 --Twarm 20 --weight_balance 0.5 --consistency_mode feature_only --eval_pseudo > FedpuCIFARFeatureWarm20.log 2>&1 &
 
nohup python -u main.py --model fedpu --dataset fedpu_imagenette --device_id 7 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 0 1 2 8 9 --public_size 100 --label_freq 0.2 --Twarm 20 --weight_balance 0.5 --consistency_mode feature_only --eval_pseudo > FedpuImagenetteFeatureWarm20.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_usps --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 4 7 5 8 9 --public_size 100 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode feature_only --eval_pseudo > FedpuUSPSFeatureWarm30.log 2>&1 &

nohup python -u main.py --model fedpu --dataset fedpu_letter --device_id 2 --local_lr 0.01 --communication_epoch 100 --local_epoch 5 --local_batch_size 64 --parti_num 50 \
 --beta 0.75 --online_ratio 0.2 --pos_class_list 1 21 11 17 8 14 22 18 9 10 2 7 25 --public_size 260 --label_freq 0.2 --Twarm 30 --weight_balance 0.5 --consistency_mode feature_only --eval_pseudo > FedpuLetterFeatureWarm30.log 2>&1 &