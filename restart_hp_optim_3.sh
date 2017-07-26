#!/usr/bin/env

START=1
END=1000
for (( i=$START; i <= $END; ++i ))
do
    CUDA_VISIBLE_DEVICES=2 python per_script_optimization/db_fig_bw.py
    CUDA_VISIBLE_DEVICES=2 python per_script_optimization/db_fig_tbp.py
    CUDA_VISIBLE_DEVICES=2 python per_script_optimization/db_fig_tbtcso.py
done

