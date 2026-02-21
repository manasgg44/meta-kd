#!/bin/bash

python train_student_meta.py \
  --print_freq 100 \
  --tb_freq 500 \
  --save_freq 40 \
  --batch_size 64 \
  --num_workers 8 \
  --epochs 240 \
  --init_epochs 30 \
  --lr 0.05 \
  --teacher_lr 0.05 \
  --lr_decay_epochs "150,180,210" \
  --lr_decay_rate 0.1 \
  --weight_decay 0.0005 \
  --momentum 0.9 \
  --loss_type "mse" \
  --held_size 5000 \
  --num_held_samples 100 \
  --num_meta_batches 1 \
  --assume_s_step_size 0.05 \
  --dataset "cifar100" \
  --model_s "resnet8" \
  --model_t "resnet32x4" \
  --path_t "save/models/resnet32x4_vanilla/ckpt_epoch_240.pth" \
  --kd_T 4 \
  --trial "1" \
  --alpha 0.5