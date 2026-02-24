#!/bin/bash

python train_student_meta_ta.py \
  --dataset cifar100 \
  --model_s resnet32 \
  --model_t resnet110 \
  --path_t ./save/models/resnet110_vanilla/ckpt_epoch_240.pth \
  --loss_type kl --kd_T 4 \
  -a 0.5 -b 0.5 \
  --held_size 5000 --num_held_samples 100 --num_meta_batches 1 \
  --assume_s_step_size 0.05