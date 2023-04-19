# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 13:05:54 2022

@author: aarregui
"""

import os
import pickle
import pandas as pd
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (20,15)
from src.utils import directories
import seaborn as sns

analysis = ['ARIMA', 'GP', 'LRVAR', 'BVAR', 'MTGP']

univariate = ['ARIMA','GP']
multivariate = ['LRVAR', 'BVAR', 'MTGP']


output_folder = os.path.join('output')
predictions_folder = 'predictions'
general_results_folder = 'general_results'


all_cells = ['b1c0', 'b1c1', 'b1c2', 'b1c3', 'b1c4', 'b1c5', 'b1c6', 'b1c7',
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


n_instants_array = np.array([50, 100, 150])

variables = ['T','Qd', 'Qc', 'I', 'V']

variables_human = ['Temperature', 'Discharge Capacity', 'Charging Capacity', 
                   'Internal Resistance', 'Voltage']

no_dir = list()
general_results = os.path.join(output_folder, general_results_folder, 'boxplots')
directories.validate_directory(general_results)

#%%


plt.rcParams.update({'font.size': 20})

error = 'NRMSE'

analysis_true = ['ARIMA', 'GP', 'RRVAR', 'BVAR', 'MTGP']

results_folder = os.path.join(output_folder, 'general_results')

files = os.listdir(results_folder)
cells = [file for file in files if 'pickle' in file]
cells = [cell for cell in cells if cell.split('_')[0] in all_cells]

df_all_instants = pd.DataFrame([])

for n_instants in n_instants_array:
    
    res_n_instants = pd.DataFrame([])
    
    df_instants = pd.DataFrame([])
    
    for index, var in enumerate(variables):
        
        df_all_cells = pd.DataFrame([])
        
        cells_prac = [file for file in cells if '_{}_T{}'.format(var, n_instants) in file]        
        
        for file in cells_prac:
            
            cell_name = file.split('_')[0]
            file_name = os.path.join(results_folder, file)
            
            if not os.path.isfile(file_name):
                no_dir.append(file_name)
                continue
            
            with open(file_name, 'rb') as handle:
                errors_dict = pickle.load(handle)
            
            df_cell = pd.DataFrame([])
            try:
                for an in analysis:
                    df_cell[an] = errors_dict[an][error]
            except:
                continue
            
            df_cell['cell'] = cell_name
                
            
            if df_all_cells.empty:
                df_all_cells = df_cell.copy()
            else:
                df_all_cells = pd.concat([df_all_cells, df_cell], ignore_index = True)
                
                
        df_all_cells = df_all_cells.replace([np.inf, -np.inf], np.nan).dropna()
        #compute statistics
        df_stats = df_all_cells[analysis].copy()
        df_stats = df_stats.loc[(df_stats>1).sum(axis = 1) == 0]
        res_file_name = os.path.join(general_results, 'stats_T{}.csv'.format(n_instants))
        res_var = df_stats.apply(lambda x: pd.Series({'mean_{}'.format(var): x.mean(), 
                                                              'var_{}'.format(var): x.var(),
                                                              'median_{}'.format(var): x.median(), 
                                                              'IQR_{}'.format(var): np.subtract(*np.percentile(x, [75, 25]))})).T
        
        res_var = res_var.reset_index()
        res_var = res_var.rename(columns = {'index': 'analysis'})
        
        if res_n_instants.empty:
            res_n_instants = res_var.copy()
        else:
            res_n_instants = res_n_instants.merge(res_var, how = 'inner')
        
        
        #update df_instants
        for an in analysis:
            
            df_to_append = pd.DataFrame([])
            df_to_append['error'] = df_all_cells[an].to_numpy()
            df_to_append['model'] = an
            df_to_append['var'] = var
            
            if df_instants.empty:
                
                df_instants = df_to_append.copy()
                
            else:
                
                df_instants = pd.concat([df_instants, df_to_append], ignore_index = True)
        
        #stats raw
        df_bycell = df_all_cells.groupby('cell').mean()
        df_bycell = df_bycell.add_suffix('_mean')
        
        
        df_bycell_toappend = df_all_cells.groupby('cell').std()
        df_bycell_toappend = df_bycell_toappend.add_suffix('_std')
        df_bycell = pd.concat([df_bycell, df_bycell_toappend], axis = 1)
        
        df_bycell_toappend = df_all_cells.groupby('cell').median()
        df_bycell_toappend = df_bycell_toappend.add_suffix('_median')
        df_bycell = pd.concat([df_bycell, df_bycell_toappend], axis = 1)
    
        df_bycell_toappend = df_all_cells.groupby('cell').agg(iqr)
        df_bycell_toappend = df_bycell_toappend.add_suffix('_IQR')
        df_bycell = pd.concat([df_bycell, df_bycell_toappend], axis = 1)
        
        #reset index
        df_bycell = df_bycell.reset_index()
        df_bycell_name = os.path.join(general_results, 'stats_bycell_{}_T{}.csv'.format(var, n_instants))
        df_bycell.to_csv(df_bycell_name, index = False)
    
        df_all_name = os.path.join(general_results, 'raw_results_{}_T{}.csv'.format(var, n_instants))
        df_all_cells.to_csv(df_all_name, index = False)
    
        df_cells_name = os.path.join(general_results, 'raw_results_{}_T{}.csv'.format(var, n_instants))
        df_all_cells.to_csv()
    
    #store statistics
    res_n_instants.to_csv(res_file_name, index = False)
    
    df_instants['T'] = n_instants
    if df_instants.empty:
        df_all_instants = df_instants.copy()
    else:
        df_all_instants = pd.concat([df_all_instants, df_instants], ignore_index = True)
    

df_all_instants = df_all_instants.loc[df_all_instants['error'] < 100]


#%%

sns.set_theme(style='white', font_scale = 2)
#plot using seaborn

fig, axes = plt.subplots(5, 1, figsize=(20, 15), sharex=True)


for index, var in enumerate(variables):
    
    df_plot = df_all_instants.loc[df_all_instants['var'] == var]
    
    if index != 2:
        bp = sns.boxplot(ax = axes[index], y = 'error', x = 'model', data = df_plot,
                         palette = 'Blues', hue = 'T', showfliers = False, meanline = True)
        bp.legend_.remove()
        bp.set(xlabel = None, ylabel = None)
        
    else:
        bp = sns.boxplot(ax = axes[index], y = 'error', x = 'model', data = df_plot,
                         palette = 'Blues', hue = 'T', showfliers = False, meanline = True)
        
        sns.move_legend(bp, bbox_to_anchor=(1.02, 1), loc='upper left')        
        handles, labels = axes[index].get_legend_handles_labels()
        axes[index].legend(handles=handles, labels=true_labels)
        sns.move_legend(bp, bbox_to_anchor=(1, 0.9), loc='upper left')
        bp.set(xlabel = None, ylabel = None)
    
    axes[index].set_title(variables_human[index])
        
axes[index].set_xticklabels(analysis_true)

plt.subplots_adjust(top = 1.5)

svg_file_name = os.path.join(general_results, 'boxplot.svg')
png_file_name = os.path.join(general_results, 'boxplot.png')

fig.tight_layout()
fig.savefig(png_file_name)
plt.show()


#%%



multivariate_true = ['RRVAR', 'BVAR', 'MTGP']
true_labels = ['$T = 50$', '$T = 100$', '$T = 150$']
#plot using seaborn

fig, axes = plt.subplots(5, 1, figsize=(20, 15), sharex=True)

df_md = df_all_instants.loc[df_all_instants['model'].isin(multivariate)]

for index, var in enumerate(variables):
    
    df_plot = df_md.loc[df_md['var'] == var]
    
    
    if index != 2:
        bp = sns.boxplot(ax = axes[index], y = 'error', x = 'model', data = df_plot,
                         palette = 'Blues', hue = 'T', showfliers = False, meanline = True)
        bp.legend_.remove()
        bp.set(xlabel = None, ylabel = None)
        
    else:
        bp = sns.boxplot(ax = axes[index], y = 'error', x = 'model', data = df_plot,
                         palette = 'Blues', hue = 'T', showfliers = False, meanline = True)
        
        
        handles, labels = axes[index].get_legend_handles_labels()
        axes[index].legend(handles=handles, labels=true_labels)
        sns.move_legend(bp, bbox_to_anchor=(1, 0.9), loc='upper left')
        bp.set(xlabel = None, ylabel = None)
    
    axes[index].set_title(variables_human[index])
        
axes[index].set_xticklabels(multivariate_true)

plt.subplots_adjust(top = 1.5)

svg_file_name = os.path.join(general_results, 'boxplot_md.svg')
png_file_name = os.path.join(general_results, 'boxplot_md.png')

fig.tight_layout()
fig.savefig(png_file_name)
plt.show()


#%%

#store numerical data
df_all_instants['model'] = df_all_instants['model'].replace('LRVAR', 'RRVAR')

df_grouped = df_all_instants.groupby(['model','var', 'T']).agg(mean_error = ('error', 'mean'), var_error = ('error', 'var'))

df_grouped = df_grouped.reset_index()



for n_instants in n_instants_array:
    
    df_stats = pd.DataFrame([])
    
    for var in variables:
        df_to_append = df_grouped.loc[((df_grouped['T'] == n_instants)&(df_grouped['var'] == var))].copy()
        df_to_append = df_to_append.drop(['var', 'T'],axis = 1)
        df_to_append['model'] = pd.Categorical(df_to_append['model'], categories=analysis_true, ordered=True)
        
        df_to_append = df_to_append.sort_values('model')
        
        df_to_append = df_to_append.rename(columns = {'mean_error':'mean_error_{}'.format(var), 'var_error':'var_error_{}'.format(var)})
        metrics_name = ['mean_error_{}'.format(var), 'var_error_{}'.format(var)]
            
        df_to_append = df_to_append.set_index('model')
        
        #fill df_stats
        if df_stats.empty:
        
            df_stats = df_to_append.copy()
        
        else:
            
            df_stats = pd.concat([df_stats, df_to_append], axis = 1)

    df_stats = df_stats.reset_index()
    file_name = os.path.join(general_results, 'stats_T{}.csv'.format(n_instants))
    df_stats.to_csv(file_name, index = False)

