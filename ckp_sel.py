# -*- coding: utf-8 --
"""
Created on Wed Sep  4 09:48:16 2019

@author: ronglai.zuo
"""

import os

dir_name = 'checkpoints/originalFFN/basic11'
results_file = 'results/test_sample/ckpsel11.txt'
models = os.listdir(dir_name)
num = len(models)
models.sort()

if num >= 10:
    for i in range(-2,-12,-1):
        model_name, _ = models[i].split('.')
        with open(results_file, 'a+') as f:
            f.write(model_name + ':\n')
        print(model_name+'...')
        os.system('python inference.py --model_name '+model_name)
else:
    print('num is not enough')
