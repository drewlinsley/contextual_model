#!/bin/bash
cd /home/drew/Documents/tf_experiments/experiments/contextual_circuit
CUDA_VISIBLE_DEVICES=2 python optimize_fig_3.py --lesion T Q
CUDA_VISIBLE_DEVICES=2 python optimize_fig_3.py --lesion T P
CUDA_VISIBLE_DEVICES=2 python optimize_fig_3.py --lesion T U
CUDA_VISIBLE_DEVICES=2 python optimize_fig_3.py --lesion Q P
