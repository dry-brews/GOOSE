#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 18:04:27 2021

@author: bryan
"""

import sys
import random
from math import floor

fraction_testing = 0.10

with open(sys.argv[1],'r') as scores_file:
    lc = 0
    all_lines = []
    for line in scores_file:
        lc +=1
        if lc == 1:
            header = line
        else:
            all_lines.append(line)

random.shuffle(all_lines)
test_count = floor(fraction_testing * len(all_lines))

with open("testing_set.tsv",'w+') as testing_file:
    testing_file.write(header)
    for i in range(0,test_count):
        testing_file.write(all_lines[i])
    
with open("training_set.tsv", 'w+') as training_file:
    training_file.write(header)
    for i in range(test_count,len(all_lines)):
        training_file.write(all_lines[i])
        