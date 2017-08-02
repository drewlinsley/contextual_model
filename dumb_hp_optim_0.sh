#!/usr/bin/env

START=1
END=1000
for (( i=$START; i <= $END; ++i ))
do
    CUDA_VISIBLE_DEVICES=0 python per_script_optimization/db_fig_3a.py
    CUDA_VISIBLE_DEVICES=0 python per_script_optimization/db_fig_3b.py
    CUDA_VISIBLE_DEVICES=0 python per_script_optimization/db_fig_4.py
done

