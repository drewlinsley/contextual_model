import os
import numpy as np
import itertools

def write_script(lines,absolute_path,out_dir):
    with open(out_dir,'w') as f:
         f.write('#!/bin/bash\n')
         f.write('cd ' + absolute_path + '\n')
         for l in lines:
             f.write('%s\n' % l)

available_gpus = [0,2,3]
lesionable_parts = ['T','Q','P','U']
absolute_path = '/home/drew/Documents/tf_experiments/experiments/contextual_circuit'
out_dir = 'scripts'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

listed_combs = [list(itertools.combinations(lesionable_parts,x)) for x in range(1,len(lesionable_parts))]
lesion_combs = []
for idx in listed_combs:
    [lesion_combs.append(x) for x in idx]

splits = np.repeat(np.array(available_gpus),len(lesion_combs)/len(available_gpus))
splits = np.append(splits,np.repeat(available_gpus[-1],len(lesion_combs)-len(splits)))
prev_gpu = 0
script = []
for gpu, lesions in zip(splits,lesion_combs):
    if gpu != prev_gpu:
	#write a new script
        write_script(script,absolute_path,os.path.join(out_dir,'gpu_'+str(prev_gpu)+'.sh'))
	script = []
    script.append('CUDA_VISIBLE_DEVICES='+str(gpu) + ' python optimize_fig_3.py --lesion ' + ' '.join(lesions))
    prev_gpu = gpu
write_script(script,absolute_path,os.path.join(out_dir,'gpu_'+str(gpu)+'.sh'))
