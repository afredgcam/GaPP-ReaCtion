# view results

import numpy as np
import functions as f
import matplotlib.pyplot as plt
from tqdm import tqdm

# plot design choices
plt.rcParams["figure.dpi"] = 500
plt.rcParams['text.usetex'] = True
colours = ['blueviolet','mediumvioletred','turquoise','teal','deeppink',
           'indigo','fuchsia','mediumspringgreen','deepskyblue']

# choose data sets (indexed 0-99)
data_sets = np.arange(100)
num_sets = len(data_sets)

# prepare metrics (subset of ['GaPP-Class','GaPP-ReaCtion','DiGiT','MP-IMM','MP-CV','GM-PHD','GNN-CV'])
methods = ['GaPP-ReaCtion','GaPP-Class','DiGiT','MP-IMM','MP-CV','GM-PHD','GNN-CV']
num_methods = len(methods)
metric_names = ['C','A','S','PA','mR','GOSPA','s2','γ','µ0','µi','Ci'] # mR = milli-R = R/1000
num_std_metrics = 6
num_metrics = len(metric_names)
metrics = np.full([num_sets,num_methods,num_metrics],np.nan)

# choices
wanting_main_plot = False
scale_factor = 1


for set_ind in tqdm(range(num_sets)):
    
    # load data
    set_num = data_sets[set_ind]
    file = np.load(f'Data/set{set_num}.npz',allow_pickle=True)
    y,truths = list(file['data']),list(file['tracks'])
    true_births,true_deaths = file['births'],file['deaths']
    S,aspect,dt = file['scene'],file['aspect'],list(file['parameters'])[1]
    MQ = [truths,true_births,dt]
    aspect = [asp*scale_factor for asp in aspect]
    
    for m in range(num_methods):
        # unpack results for this method / data set
        tracks_m,births_m,metrics_m = f.open_results(methods[m],set_num,MQ)
        metrics[set_ind,m,:metrics_m.shape[0]] = metrics_m
        # plot, if desired
        if wanting_main_plot:
            deaths_m = np.array([births_m[i] + len(tracks_m[i])
                                 for i in range(len(tracks_m))],int)
            f.plot_results(y,truths,true_deaths,tracks_m,deaths_m,S,colours,
                           methods[m],aspect,False)

# view metrics
f.value_table(methods,metric_names,metrics,rounder=2)