# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 12:36:17 2023

@author: aarregui
"""

import os
import pandas as pd
import numpy as np
from src.utils import directories
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (30,15)
import seaborn as sns


analysis = ['mean', 'ARIMA', 'GP', 'BVAR', 'LRVAR', 'MTGP']
analysis = ['ARIMA', 'GP', 'BVAR', 'LRVAR', 'MTGP']


univariate = ['ARIMA','GP']
multivariate = ['BVAR', 'LRVAR' 'MTGP']
naive = ['mean']

output_folder = os.path.join('output')
predictions_folder = 'predictions'
training_time_folder = os.path.join(output_folder, 'general_results','training_time')
directories.validate_directory(training_time_folder)

n_instants_array = np.array([50,100,150])


variables = ['Qd', 'Qc', 'I', 'V', 'T']
variables_human = ['Discharge capacity', 'Charging capacity', 
                   'Internal Resistance', 'Voltage', 'Temperature']
analysis_true = ['AVG','ARIMA', 'GP', 'BVAR', 'RRVAR', 'MTGP']
analysis_true = ['ARIMA', 'GP', 'BVAR', 'RRVAR', 'MTGP']

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

sns.set_theme(style='white', font_scale = 2)


general_results = os.path.join('output','general_results')

df_all = pd.DataFrame([])
fig, axes = plt.subplots(3, 1, figsize=(20, 15), sharex=True)
for index,n_instants in enumerate(n_instants_array):
    
    training_file_name = os.path.join(general_results,'training_time', 'training_T{}.csv'.format(n_instants))
    
    df = pd.read_csv(training_file_name, index_col = 0)
    

    col_list = df.columns.tolist()
    
    df_wide_var = pd.DataFrame([])
    
    for var in variables:
        
        col_var = (elem for elem in df.columns.tolist() if var in elem)
        
        df_plot = df[col_var].copy()

        #from wide to long
        df_wide = pd.DataFrame([])
        for an in analysis:
            col_name = '{}_{}'.format(an, var)
            
            df_to_append = pd.DataFrame({'Time (s)' : df_plot[col_name].to_numpy()})
            df_to_append['model'] = an
            
            df_to_append.index = df_plot.index.values
            
            if df_wide.empty:

                df_wide = df_to_append.copy()
            else:
                
                df_wide = pd.concat([df_wide, df_to_append])
        
        df_wide.index.name = 'Cycles'
        
        df_wide = df_wide.reset_index()
        
        df_wide['model'] = df_wide['model'].replace('LRVAR','RRVAR')
        df_wide['model'] = df_wide['model'].replace('mean', 'AVG')
        df_wide['var'] = var
        if df_wide_var.empty:
            df_wide_var = df_wide.copy()
        else:
            df_wide_var = pd.concat([df_wide_var, df_wide], ignore_index = True)
        
        
    df_wide_var['T'] = n_instants
    
    
    if df_all.empty:
        df_all = df_wide_var.copy()
    else:
        df_all = pd.concat([df_all, df_wide_var], ignore_index = True)

    if index != 1:
        line = sns.lineplot(ax = axes[index], data = df_wide_var, x = "Cycles", y = "Time (s)", hue = 'model', style = 'model', lw = 5, legend = False, ci = None)
        
    else:
        line = sns.lineplot(ax = axes[index], data = df_wide_var, x = "Cycles", y = "Time (s)", hue = 'model', style = 'model', lw = 5, legend = 'full', ci = None)
               
        leg = axes[index].legend(bbox_to_anchor=(1.04, 0.5), loc="center left")
        
        for line in leg.get_lines():
            line.set_linewidth(4.0)
            
        
    axes[index].set(yscale = 'log')
    axes[index].set_ylabel('')
    #grid lines
    axes[index].grid(True, which="minor", axis = 'y', ls="--", c='gray', alpha = .3)  
    axes[index].grid(True, which = 'major', axis = 'y', ls = '-', c = 'gray', alpha = .6)
    axes[index].set_title('Cycle length $T = {}$'.format(n_instants))
            
    
plt.subplots_adjust(top = 15)

svg_file_name = os.path.join(general_results,'training_time', 'TT.svg')
png_file_name = os.path.join(general_results, 'training_time', 'TT.png')

fig.tight_layout()
fig.savefig(svg_file_name)
fig.savefig(png_file_name)
plt.show()


#%%


df_agg = df_all[['Cycles','Time (s)', 'model', 'T']].copy()

df_to_store = df_agg.groupby(['Cycles', 'model', 'T']).mean()

df_to_store = df_to_store.reset_index()[['Time (s)', 'model', 'T']].copy()

df_to_store = df_to_store.groupby(['model', 'T']).mean().reset_index()

    
df_new = pd.DataFrame([])

for n_instants in n_instants_array:
    
    new_col = 'T = {}'.format(n_instants)
    df_filt = df_to_store.loc[df_to_store['T']==n_instants][['model','Time (s)']]
    
    df_filt = df_filt.rename({'Time (s)': new_col}, axis = 1)
        
    df_filt = df_filt.set_index('model')
    if df_new.empty:
        df_new = df_filt.copy()
    else:
        df_new = pd.concat([df_new, df_filt], axis = 1)


df_new = df_new.reindex(analysis_true)

agg_file_name = os.path.join(general_results,'training_time','TT.csv')
#df_new.to_csv(agg_file_name)

