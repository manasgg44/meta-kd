#!/bin/bash

python test_teacher.py \
	--teacher resnet32x4 \
	--save-folder './save/student_model/S:resnet8_T:resnet32x4_cifar100_mlkd_a:None_1' \
	--checkpoint "${CHECKPOINT:-}" \
	--batch-size 64 \
	--num-workers 8 \
	--print-freq 100
