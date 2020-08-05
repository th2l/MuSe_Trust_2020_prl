#!/bin/bash

export PYTHONHASHSEED=0
trap "exit" INT
sleep 1
dm_str=$(date +'%Y_%m_%d')
tm_str=$(date +'%H_%M_%S')

python src/main.py --dir ./train_logs/v20/"$dm_str"_"$tm_str" \
        --test ''	\
        --task 3 \
        --seed 1 \
        --epochs 50 \
        --gaussian_noise 0.01 \
        --lr_init 1e-2 \
        --use_min_lr 41 \
        --min_lr 1e-3 \
        --batch_size 32 \
        --opt sgd \
        --use_data raw_audio fasttext landmarks_2d
sleep 1
#dm_str=$(date +'%Y_%m_%d')
#tm_str=$(date +'%H_%M_%S')
#python src/main.py --dir ./train_logs/v19/"$dm_str"_"$tm_str" \
#        --test ''	\
#        --task 3 \
#        --seed 1 \
#        --epochs 50 \
#        --gaussian_noise 0.01 \
#        --lr_init 1e-2 \
#        --use_min_lr 41 \
#        --min_lr 1e-3 \
#        --batch_size 128 \
#        --opt sgd \
#        --use_data deepspectrum fasttext vggface
#sleep 1
#dm_str=$(date +'%Y_%m_%d')
#tm_str=$(date +'%H_%M_%S')
#python src/main.py --dir ./train_logs/v19/"$dm_str"_"$tm_str" \
#        --test ''	\
#        --task 3 \
#        --seed 1 \
#        --epochs 50 \
#        --gaussian_noise 0.01 \
#        --lr_init 1e-2 \
#        --use_min_lr 41 \
#        --min_lr 1e-3 \
#        --batch_size 128 \
#        --opt sgd \
#        --multitask \
#        --use_data deepspectrum fasttext landmarks_2d
#sleep 1
#dm_str=$(date +'%Y_%m_%d')
#tm_str=$(date +'%H_%M_%S')
#python src/main.py --dir ./train_logs/v19/"$dm_str"_"$tm_str" \
#        --test ''	\
#        --task 3 \
#        --seed 1 \
#        --epochs 50 \
#        --gaussian_noise 0.01 \
#        --lr_init 1e-2 \
#        --use_min_lr 41 \
#        --min_lr 1e-3 \
#        --batch_size 128 \
#        --opt sgd \
#        --multitask \
#        --use_data deepspectrum fasttext vggface
#sleep 1
#python src/main.py --dir ./train_logs/v15/"$dm_str"_"$tm_str" \
#        --task 3 \
#        --seed 1 \
#        --epochs 50 \
#        --gaussian_noise 0.0 \
#        --lr_init 1e-4 \
#        --batch_size 256 \
#        --use_weight_mtl \
#        --multitask \
#        --use_data deepspectrum fasttext landmarks_2d
#
#sleep 1
#python src/main.py --dir ./train_logs/v15/"$dm_str"_"$tm_str" \
#        --task 3 \
#        --seed 1 \
#        --epochs 30 \
#        --gaussian_noise 0.0 \
#        --lr_init 1e-3 \
#        --batch_size 128 \
#        --use_data deepspectrum fasttext landmarks_2d
#
#sleep 10
#python src/main.py --dir ./train_logs/v15/"$dm_str"_"$tm_str" \
#        --task 3 \
#        --seed 1 \
#        --epochs 30 \
#        --gaussian_noise 0.0 \
#        --lr_init 1e-3 \
#        --batch_size 128 \
#        --use_data deepspectrum landmarks_2d
#
#sleep 10
#python src/main.py --dir ./train_logs/v15/"$dm_str"_"$tm_str" \
#        --task 3 \
#        --seed 1 \
#        --epochs 30 \
#        --gaussian_noise 0.0 \
#        --lr_init 1e-3 \
#        --batch_size 128 \
#        --use_data deepspectrum vggface

#sleep 10
#python src/main.py --dir ./train_logs/v15/"$dm_str"_"$tm_str" \
#        --task 3 \
#        --seed 1 \
#        --epochs 30 \
#        --gaussian_noise 0.0 \
#        --lr_init 1e-3 \
#        --batch_size 128 \
#        --use_data deepspectrum
#
#sleep 10
#python src/main.py  --dir ./train_logs/v15/"$dm_str"_"$tm_str" \
#        --task 3 \
#        --seed 1 \
#        --epochs 30 \
#        --gaussian_noise 0.0 \
#        --lr_init 1e-3 \
#        --batch_size 128 \
#        --use_data fasttext
#
#sleep 10
#python src/main.py --dir ./train_logs/v15/"$dm_str"_"$tm_str" \
#        --task 3 \
#        --seed 1 \
#        --epochs 30 \
#        --gaussian_noise 0.0 \
#        --lr_init 1e-3 \
#        --batch_size 128 \
#        --use_data landmarks_2d
#
#sleep 10
#python src/main.py  --dir ./train_logs/v15/"$dm_str"_"$tm_str" \
#        --task 3 \
#        --seed 1 \
#        --epochs 30 \
#        --gaussian_noise 0.0 \
#        --lr_init 1e-3 \
#        --batch_size 128 \
#        --use_data vggface

#use_data = {"au": False, "deepspectrum": False, "egemaps": False, "fasttext": False, "gaze": False, "gocar": False,
#                "landmarks_2d": False, "landmarks_3d": False, "lld": False, "openpose": False, "pdm": False,
#                "pose": False, "vggface": False, "xception": False}
