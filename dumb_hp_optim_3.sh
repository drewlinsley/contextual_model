#!/usr/bin/env

START=1
END=1000
i=$START
while [[ $i -le $END ]]
do 
    CUDA_VISIBLE_DEVICES=3 python per_script_optimization/db_fig_bw.py
    CUDA_VISIBLE_DEVICES=3 python per_script_optimization/db_fig_tbp.py
    CUDA_VISIBLE_DEVICES=3 python per_script_optimization/db_fig_tbtcso.py
    i=$((i + 1))
done
