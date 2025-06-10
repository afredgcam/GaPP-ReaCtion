# GaPP-ReaCtion
Python code for the proposed methods in "Integrated Gaussian Processes for Robust and Adaptive Multi-Object Tracking" (Lydeard/Ahmad/Godsill, IEEE AES 2025/6).

**Required packages:** `numpy`, `scipy`, `pandas`, `matplotlib`, `tqdm`, `stonesoup`  

**LICENSE:** MIT License – have fun!

## Contents
`Data`: Folder containing all data sets (in Python and MATLAB formats)  
`data_generation.py`: Generates the data in `Data` folder  
`functions.py`: Contains all functions and classes used in the other `.py` files   
`gapp_class_pooled.py`: Runs (parallelised) **GaPP-Class** on chosen data sets  
`gapp_reaction_pooled.py`: Runs (parallelised) **GaPP-ReaCtion** on chosen data sets  
`gm_phd.py`: Runs the GM-PHD on chosen data sets  
`gnn_cv.py`: Runs GNN-CV on chosen data sets  
`Results`: Folder containing saved results for all methods in the above paper on all data sets  
`view_data_set.py`: Plot chosen data set  
`view_results.py`: Plot inferred tracks / tabulate metrics of chosen methods on chosen data sets  

***

### File `data_generation.py`
#### Main 'Inputs'
* True fixed hyperparameters, e.g., $\psi$, $(\sigma_c^2,\ell_c)$ $\forall$ classes $c$, etc. 
* True variable hyperparameter (uniform) *ranges*, e.g., choose $(\mu_0^\text{min},\mu_0^\text{max})$ such that $\mu_0\sim U(\mu_0^\text{min},\mu_0^\text{max})$
* `wanting_full_plots = True` plots all trajectories / data for each data set upon generation
* `wanting_each_plot = True` plots every time step for each data set
* `num_sets` is the number of sets to generate (`=100` without changing the seed in `functions.py` yields those used in the above paper) 

***

### File `functions.py`

Further details about the proposed methods and / or metrics may be found in their actual implementation, here 

**Note:** The random seed *for all* files or functions is atop this script. To recover anything, e.g., data sets, (if possible – see note below), this should be `rng = np.random.default_rng(seed=28)`

***

### File `gapp_class_pooled.py`
#### Main 'Inputs'
* Prior hyperparameters, e.g., $\alpha_+$, $\beta_+$, etc. 
* Default deletion criteria, e.g., max. number of missed detections
* `data_sets` is a list / array of indices for the sets on which inference is sought
* `wanting_main_plot = True` plots the trajectories / ground truths / data for all time steps
* `wanting_hyperpar_plots = True` computes real-time estimates and approx. uncertainties of hyperparameters, and plots them
* `wanting_set_timing = True` times how long inference takes on each data set (timing over all sets is always done)
* `saving_results = True` computes metrics and saves them and estimated trajectories  

**Notes:**  

* Hyperparameter plotting or saving require more steps than regular filtering, and so *significantly* increase run-time
* Due to parallelisation, exact replication is not possible by fixing the random seed

***

### File `gapp_reaction_pooled.py`

This is identical to `gapp_class_pooled.py` except for an additional lines for the 'revival' step

***

### File `gm_phd.py`
#### Main 'Inputs'
* `data_sets` is a list / array of indices for the sets on which inference is sought
* `saving_results = True` computes metrics and saves them and estimated trajectories 
* `wanting_main_plot = True` plots the trajectories / ground truths / data for all time steps (must have `saving_results = True`; if overwriting undesired, use temporary file name)
* Parameters and design choices, e.g., CV driving noise variance, prune threshold, etc.

***

### File `gnn_cv.py`

Same as above, except with different, model-specific, parameters

***

### File `view_data_sets.py`
#### Main 'Inputs'
* `data_sets` is a list / array of indices for the sets whose plots are sought
* `scale_factor` scales the final plot

***

### File `view_results.py`
#### Main 'Inputs'
* `data_sets` is a list / array of indices for the sets on which inferences are to be compared
* `methods` are the method names whose results are to be compared
* `wanting_main_plot = True` plots the trajectories / ground truths / data for all time steps for each method separately
* `scale_factor` scales the final plot

