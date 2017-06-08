#!/bin/bash
cd /home/drew/Documents/tf_experiments/experiments/contextual_circuit
CUDA_VISIBLE_DEVICES=3 python optimize_fig_3.py --lesion Q U
CUDA_VISIBLE_DEVICES=3 python optimize_fig_3.py --lesion P U
CUDA_VISIBLE_DEVICES=3 python optimize_fig_3.py --lesion T Q P
CUDA_VISIBLE_DEVICES=3 python optimize_fig_3.py --lesion T Q U
CUDA_VISIBLE_DEVICES=3 python optimize_fig_3.py --lesion T P U
CUDA_VISIBLE_DEVICES=3 python optimize_fig_3.py --lesion Q P U
