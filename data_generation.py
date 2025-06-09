# GaPP-Class / GaPP-ReaCtion data generation

import numpy as np
import functions as f
import matplotlib.pyplot as plt
from tqdm import tqdm

# plot design choices
plt.rcParams["figure.dpi"] = 500
plt.rcParams['text.usetex'] = True

# parameters
d = 10 # iSE independence time (step) difference
dt = 1 # time step size
Tmax = 100 # total no. of time steps
s2_range = [0.5,2] # min / max observation variance
ψ = 0.98 # survival probability
γ_range = [0.02,0.1] # E(no. new objects / time step)
µ0_range = [10,15] # min / max E(clutter rate)
Nc = 2 # number of object classes
class_pars = np.array([[100,4],[10,1]]) # [σ2_c,ell_c] for all c
µi_range = [3,6] # min / max E(detection rate)

# plotting quantities
wanting_full_plots = False
wanting_each_plot = False
colours = ['blueviolet','pink','deeppink','indigo','mediumvioletred',
            'goldenrod','turquoise','teal','mediumspringgreen']
style = '-'
# first marker is start, second is end of surviving track, third is end of dead track
mrks = ['x','.','+'] 

# number of sets (num_sets = 100 recovers those used for the paper, provided)
num_sets = 100

# data sets
for set_num in tqdm(range(num_sets)):
    # generate data set
    data_set = f.DataSet(Tmax,dt,d,s2_range,ψ,γ_range,µ0_range,Nc,class_pars,
                         µi_range)
    data_set.generate()
    # save data set to folder 'Data'
    name = f'Data/set{set_num}.npz'
    data_set.save_data(name)
    if wanting_each_plot:
        for k in range(Tmax):
            # make images for GIF of scenario
            data_set.plot_step(k,colours,style,mrks)
    if wanting_full_plots:
        # show whole data set
        data_set.plot(colours,style,mrks)