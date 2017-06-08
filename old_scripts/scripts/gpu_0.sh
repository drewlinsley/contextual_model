#!/bin/bash
cd /home/drew/Documents/tf_experiments/experiments/contextual_circuit
CUDA_VISIBLE_DEVICES=0 python optimize_fig_3.py --lesion T
CUDA_VISIBLE_DEVICES=0 python optimize_fig_3.py --lesion Q
CUDA_VISIBLE_DEVICES=0 python optimize_fig_3.py --lesion P
CUDA_VISIBLE_DEVICES=0 python optimize_fig_3.py --lesion U
