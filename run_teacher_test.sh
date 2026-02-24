#!/usr/bin/env bash

python test_teacher.py \
  --teacher "resnet32x4" \
  --save_folder "./save/student_model/S:resnet8_T:resnet32x4_cifar100_mlkd_a:None_1" \
  --checkpoint "" \
  --batch_size 64 \
  --num_workers 8 \
  --print_freq 100