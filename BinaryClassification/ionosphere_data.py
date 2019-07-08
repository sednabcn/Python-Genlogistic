#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 18:17:30 2018

@author: sedna
"""

import pandas as pd

ionosphere_all_data=pd.read_csv('ionosphere_data_kaggle.csv',index_col=0)
print(ionosphere_all_data.head())
y=ionosphere_all_data.pop('label').to_frame()
X=ionosphere_all_data
print(X.head())
print(y.head())