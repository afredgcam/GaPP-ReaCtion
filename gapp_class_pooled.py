# parallelised GaPP-Class

import numpy as np
import matplotlib.pyplot as plt
import functions as f
import time
from multiprocessing import Pool
from functools import partial

# plot design choices
plt.rcParams["figure.dpi"] = 500
plt.rcParams['text.usetex'] = True
colours = ['blueviolet','mediumvioletred','turquoise','teal','deeppink',
           'indigo','pink','mediumspringgreen']

# inference parameters
α0p = 9 # mean 12, variance 16
β0p = 3/4
αp = 4 # mean 4, variance 4
βp = 1
εp = 0.05 # mean 0.05, variance 0.05
ξp = 1
φp = 3 # mean 1, variance 1
bp = 2

# default deletion criteria
dζ = 3 # ≤ zero_count_limit
rate_limit = 0.5
zero_count_limit = 3
sd_limit = 50

# choose data sets (indexed 0-99)
data_sets = np.arange(100)

# other choices
wanting_main_plot = True
wanting_hyperpar_plots = True
wanting_set_timing = True
saving_results = True
all_metrics = np.zeros([len(data_sets),11])

# timing
global_start = time.time()





for set_num in data_sets:
    
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
    Tmax,dt,d,s2,ψ,γ,µ0,Nc,class_pars,µi_range = list(file['parameters'])
    ρ = (S[:,1]-S[:,0]).prod() ** -1
    
    # compile properties
    particle_props = [ψ,α0p,β0p,εp,ξp,φp,bp]
    track_props = [d,dζ,dt,Nc,class_pars,αp,βp,rate_limit,zero_count_limit,
                  sd_limit]
    scene_props = [ρ,S,aspect]
    
    # initialise particle filter
    J = 50 # number of particles
    pf = f.ParticleFilter(J,particle_props,track_props,scene_props)
    
    # timing
    set_start = time.time()
    
    # allow multiprocessing
    if __name__ == '__main__':
        with Pool() as p:
    
            # make inferences
            for k in range(Tmax):
                
                # resample
                if k != 0:
                    pf.resample()
                
                # data / clusters at k
                yk = y[k]
                Gk = pf.clustering(yk)
                
                # univariate iteration
                iter_part = partial(f.iterate,y=yk,G=Gk,k=k)
                
                # survival / predict / association / update steps
                iterated_particles = p.map(iter_part,pf.particles)
                pf.particles = iterated_particles
                
                # compute weights
                pf.update_w(yk)
                
                # save filtering quantities (NOTE: slower with this)
                if wanting_hyperpar_plots or saving_results:
                    pf.save_filter()
                    pf.save_full()
                    pf.track_to_truth_k(k,truths,true_births)
            
    # timing
    set_end = time.time()
    if wanting_set_timing and __name__ == '__main__':
        print(f'time for set {set_num}: ' + f.time_taken(set_start,set_end))
    
    # plotting
    if wanting_main_plot and __name__ == '__main__':
        pf.plot_scenario(truths,true_deaths,y,colours)
        print('\nTrue Start Locations:')
        for i in range(len(truths)):
            print(f'{i+1}: {truths[i][0,:]}')
    if wanting_hyperpar_plots and __name__ == '__main__':
        pf.plot_global_hyperpars(s2,γ,µ0)
        pf.plot_local_hyperpars(truths,true_µis,true_classes,true_births)
    
    # saving / metrics
    if saving_results and __name__ == '__main__':
        set_ind = np.where(data_sets == set_num)[0][0]
        name = f'Results/GaPP_Class{set_num}.npz'
        ground_truths = [truths,true_births,true_deaths,s2,γ,µ0,true_µis,
                         true_classes]
        all_metrics[set_ind,:] = pf.save(name,ground_truths)

# timing
global_end = time.time()
if __name__ == '__main__':
    print('\n–––––')
    print('Total Time: ' + f.time_taken(global_start,global_end))

# metrics to view here, if desired
avg_metrics = all_metrics.mean(0)