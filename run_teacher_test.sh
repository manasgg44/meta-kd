#!/usr/bin/env bash

python test_teacher.py \
  --teacher "resnet110" \
  --save_folder "./save/student_model/S:resnet32_T:resnet110_cifar100_mlkd_a:0.5_1/resnet110_teacher_last.pth" \
  --checkpoint "" \
  --batch_size 64 \
  --num_workers 8 \
  --print_freq 100