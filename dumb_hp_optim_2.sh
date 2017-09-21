#!/usr/bin/env

START=1
END=1000
for (( i=$START; i <= $END; ++i ))
do
    CUDA_VISIBLE_DEVICES=3 python per_script_optimization/db_fig_5.py
    CUDA_VISIBLE_DEVICES=3 python per_script_optimization/db_fig_7.py
done


