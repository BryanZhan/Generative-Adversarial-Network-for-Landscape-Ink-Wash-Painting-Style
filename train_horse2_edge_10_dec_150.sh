#!/bin/bash
# the consistency weight for each pyramid is decreas
CUDA_VISIBLE_DEVICES=0 python train.py --dataroot   ./scenery  --name 111 --model cycle_gan --no_dropout --display_port 8097 --display_freq 10 --lambda_sup 10 --lambda_A 10 --lambda_B 10 --start_dec_sup 150 --lambda_ink 0.05 --lambda_Geom 0.5 --lambda_recog 0.5 --loadSize 286 --fineSize 256 --feats2Geom_path 'checkpoints/feats2Geom/feats2depth.pth' --batchSize 1 --which_epoch 200
