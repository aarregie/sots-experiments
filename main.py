# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:28:10 2022

@author: aarregui
"""

import sys
import numpy as np
from src.pipeline_naive_last import main_naive_last
from itertools import product

#cell_idx, var_idx, instants_idx = np.array(sys.argv)[1:].astype(int)-1

cell_idx, var_idx, instants_idx = np.array([0,0,0])

cells = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4', 'b1c5', 'b1c6', 'b1c7',
       'b1c8', 'b1c9', 'b1c10', 'b1c11', 'b1c12', 'b1c13', 'b1c14',
       'b1c15', 'b1c16', 'b1c17', 'b1c18', 'b1c19', 'b1c20', 'b1c21',
       'b1c22', 'b1c23', 'b1c24', 'b1c25', 'b1c26', 'b1c27', 'b1c28',
       'b1c29', 'b1c30', 'b1c31', 'b1c32', 'b1c33', 'b1c34', 'b1c35',
       'b1c36', 'b1c37', 'b1c38', 'b1c39', 'b1c40', 'b1c41', 'b1c42',
       'b1c43', 'b1c44', 'b1c45', 'b2c1', 'b2c3', 'b2c4', 'b2c5', 'b2c6',
       'b2c10', 'b2c11', 'b2c12', 'b2c13', 'b2c14', 'b2c17', 'b2c18',
       'b2c19', 'b2c20', 'b2c21', 'b2c22', 'b2c23', 'b2c24', 'b2c25',
       'b2c26', 'b2c27', 'b2c28', 'b2c29', 'b2c30', 'b2c31', 'b2c32',
       'b2c33', 'b2c34', 'b2c35', 'b2c36', 'b2c37', 'b2c38', 'b2c39',
       'b2c40', 'b2c41', 'b2c42', 'b2c43', 'b2c44', 'b2c45', 'b2c46',
       'b2c47', 'b3c0', 'b3c1', 'b3c3', 'b3c4', 'b3c5', 'b3c6', 'b3c7',
       'b3c8', 'b3c9', 'b3c10', 'b3c11', 'b3c12', 'b3c13', 'b3c14',
       'b3c15', 'b3c16', 'b3c17', 'b3c18', 'b3c19', 'b3c20', 'b3c21',
       'b3c22', 'b3c24', 'b3c25', 'b3c26', 'b3c27', 'b3c28', 'b3c29',
       'b3c30', 'b3c31', 'b3c33', 'b3c34', 'b3c35', 'b3c36', 'b3c38',
       'b3c39', 'b3c40', 'b3c41', 'b3c44', 'b3c45']



variables = ['V','Qd','T','I','Qc']
instants_set = np.array([50, 100, 150])

variables, instants_set = ['V','Qd','I','Qc'], [50, 100, 150]
cell = cells[cell_idx]
cqd = variables[var_idx]
series_length = instants_set[instants_idx]

combinations = list(product(instants_set, variables, cells))

for series_length, cqd, cell in combinations:
    #NAIVE
    print(cell, cqd, series_length)
    main_naive_last(n_instants = series_length, var = cqd, cell_name = cell)

#UD
# main_GP(n_instants = series_length, var = cqd, cell_name = cell)
# main_ARIMA(n_instants = series_length, var = cqd, cell_name = cell)
# #MD
# main_BVAR(n_instants = series_length, var = cqd, cell_name = cell)
# main_LRVAR(n_instants = series_length, var = cqd, cell_name = cell)
# main_MTGP(n_instants = series_length, var = cqd, cell_name = cell)




