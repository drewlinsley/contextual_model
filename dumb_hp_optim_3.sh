#!/bin/sh
CUDA_VISIBLE_DEVICES=3 python per_script_optimization/db_fig_bw.py
CUDA_VISIBLE_DEVICES=3 python per_script_optimization/db_fig_tbp.py
CUDA_VISIBLE_DEVICES=3 python per_script_optimization/db_fig_tbtcso.py
