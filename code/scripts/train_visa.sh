#!/bin/bash

deepspeed --include localhost:2,7 --master_port 60335 train_visa.py \
    --model openllama_peft \
    --stage 1\
    --imagebind_ckpt_path ../pretrained_ckpt/imagebind_ckpt/imagebind_huge.pth\
    --image_root_path ../data/images/\
    --save_path  ./ckpt/train_visa/\
    --log_path ./ckpt/train_visa/log_rest/
