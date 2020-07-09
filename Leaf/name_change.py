#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 11:02:30 2019

@author: wzhan
"""

import os 
os.chdir('/Users/wzhan/Downloads/leaves/')

print(os.getcwd())

#i = 10000
#for f in os.listdir():
#    file_dir = os.path.join('/Users/wzhan/Downloads/leaves/', f, 'images')
#    if not os.path.exists(file_dir):
#        continue 
#    else:
#        os.chdir(file_dir)
#    for image_name in os.listdir():
#        new_name = os.path.join('/Users/wzhan/Downloads/leaves/', image_name)
##        f_name, f_ext = os.path.splitext(f)
##        f_IDs, f_format = f_name.split('_')
##        f_num = int(f_IDs[5:10]) + 10119
##        f_IDs = 'plant{}'.format(str(f_num))
##        new_name = '{}_{}{}'.format(f_IDs, f_format, f_ext)
#        os.rename(os.path.join(file_dir, image_name), new_name)

for file_name in os.listdir():
    f_name, f_ext = os.path.splitext(file_name)
    f_ID, f_format = f_name.split('_')
    new_name = '{}_rgb{}'.format(f_ID, f_ext)
    os.rename(file_name, new_name)