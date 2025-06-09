# GaPP-Class / GaPP-ReaCtion data viewer

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# plot design choices
plt.rcParams["figure.dpi"] = 500
plt.rcParams['text.usetex'] = True
colours = ['blueviolet','mediumvioletred','turquoise','teal','deeppink',
            'indigo','pink','mediumspringgreen']

# choose data sets (indexed 0-99)
data_sets = [0]

# choose plot scale factor
scale_factor = 1


for set_num in tqdm(data_sets):
    
    # load data
    file = np.load(f'Data/set{set_num}.npz',allow_pickle=True)
    y = list(file['data'])
    a = list(file['associations'])
    truths = list(file['tracks'])
    true_µis = file['Po_means']
    true_classes = file['classes']
    true_births,true_deaths = file['births'],file['deaths']
    true_status = list(file['status'])
    true_yis = list(file['yis'])
    S,aspect = file['scene'],file['aspect']
    aspect = [asp*scale_factor for asp in aspect]
    Tmax,dt,d,s2,ψ,γ,µ0,Nc,class_pars,µi_range = list(file['parameters'])
    
    # set aspect ratio
    plt.rcParams["figure.figsize"] = aspect
    
    # plot data
    for k in range(Tmax):
        name = 'Obs.' if k == 0 else ''
        plt.plot(y[k][:,0],y[k][:,1],'k2',alpha=0.2,label=name)
    for i,x in zip(range(len(truths)),truths):
        name = 'Track' if i == 0 else ''
        plt.plot(x[:,0],x[:,1],c=colours[i%len(colours)],ls='-',label=name)
        plt.plot(x[0,0],x[0,1],c=colours[i%len(colours)],ls='',marker='x',
                 label='')
        end_mk = '.' if true_deaths[i] == Tmax else '+'
        plt.plot(x[-1,0],x[-1,1],c=colours[i%len(colours)],ls='',marker=end_mk,
                 label='')
    plt.xlim(S[0,:])
    plt.ylim(S[1,:])
    # plt.legend(loc=0,framealpha=1)
    plt.show()
    
        