#!/bin/bash

export PYTHONHASHSEED=0
trap "exit" INT
sleep 1
dm_str=$(date +'%Y_%m_%d')
tm_str=$(date +'%H_%M_%S')

# Submission 2 - devel score = 0.3426, test score = 0.3259
python src/main.py --dir ./train_logs/"$dm_str"_"$tm_str" \
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
        --use_data deepspectrum fasttext landmarks_2d
sleep 1

# Submission 3 - devel score = 0.3193, test score = 0.3353
dm_str=$(date +'%Y_%m_%d')
tm_str=$(date +'%H_%M_%S')
python src/main.py --dir ./train_logs/"$dm_str"_"$tm_str" \
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
       --use_data deepspectrum fasttext vggface
