#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:17:30 2018

@author: sedna
"""

import pandas as pd

sonar_all_data=pd.read_csv('sonar.all-data.csv',index_col=0,\
                           header=None,names=range(0,60))
print(sonar_all_data.head())
y=sonar_all_data.pop(59).to_frame()
X=sonar_all_data
print(X.head())
print(y.head())