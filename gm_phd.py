# GM-PHD for synthetic data: time taken = 18:10

import numpy as np
import functions as f
from tqdm import tqdm
from datetime import datetime,timedelta
from stonesoup.models.transition.linear import \
    CombinedLinearGaussianTransitionModel as CLGTM, ConstantVelocity as CV
from stonesoup.types.detection import Detection
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.updater.pointprocess import PHDUpdater
from stonesoup.hypothesiser.gaussianmixture import GaussianMixtureHypothesiser
from stonesoup.mixturereducer.gaussianmixture import GaussianMixtureReducer
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.types.state import TaggedWeightedGaussianState as TWGS
from stonesoup.types.track import Track
from stonesoup.types.array import CovarianceMatrix
import matplotlib.pyplot as plt

# plot design choices
plt.rcParams["figure.dpi"] = 500
plt.rcParams['text.usetex'] = True
colours = ['blueviolet','mediumvioletred','turquoise','teal','deeppink',
           'indigo','pink','mediumspringgreen']

# global start time
start_time = datetime(2015,8,6,8,15)

# choose data sets (indexed 0-99)
data_sets = np.arange(100)
if len(data_sets) > 3:
    data_sets = tqdm(data_sets)
    time_sets = False
else:
    time_sets = True

# choices
saving_results = True
if saving_results:
    all_metrics = np.zeros([len(data_sets),6])
wanting_main_plot = False



for set_ind,set_num in enumerate(data_sets):
    
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
    Tmax,dt,_,_,ψ = list(file['parameters'])[:5]
    centre = S.mean(1)
    dists = S[:,1]-S[:,0]
    ρ = dists.prod() ** -1
    time_steps = tqdm(range(Tmax)) if time_sets else range(Tmax)
    times = [start_time + timedelta(microseconds=int(dt*k*1e+6))
             for k in range(Tmax)]
    metric_quantities = [truths,true_births,Tmax,dt]
    
    # parameters' prior means: from gapp_reaction_pooled.py
    µ0_prior,µi_prior,γ_prior,s2_prior = 12,4,0.05,1
    
    # other parameters
    Q = 100 # CV driving noise variance
    prune_thres = 1e-6
    merge_thres = 100
    max_comp = 100 # max no. of PHD components
    state_threshold = 0.9 # min state weight for acceptance
    birth_mean = [centre[0],0,centre[1],0] # birth component mean
    birth_covar = CovarianceMatrix(np.diag([dists[0],10,dists[1],10])**2) # birth component var 
    
    # convert data for stone soup
    measurement_model = LinearGaussian(4,(0,2),s2_prior*np.eye(2))
    all_measurements = [{Detection(y[k][l:l+1,:].T,times[k],measurement_model)
                         for l in range(len(y[k]))} for k in range(Tmax)]

    # define PHD
    transition_model = CLGTM([CV(Q),CV(Q)])
    predictor = KalmanPredictor(transition_model)
    updater = KalmanUpdater(measurement_model)
    phd_up = PHDUpdater(updater,µ0_prior*ρ,True,1-np.exp(-µi_prior),ψ)
    base_hyp = DistanceHypothesiser(predictor,updater,Mahalanobis())
    phd_hyp = GaussianMixtureHypothesiser(base_hyp,order_by_detection=True)
    reducer = GaussianMixtureReducer(prune_thres,merge_thres,max_comp)
    birth_component = TWGS(birth_mean,birth_covar,start_time,γ_prior,'birth')
    
    # prepare filtering objects
    phd_tracks = set()
    reduced_states = set()
        
    # PHD tracking
    for k in time_steps:
        yk = all_measurements[k]
        current_state = reduced_states
        birth_component.timestamp = times[k]
        current_state.add(birth_component)
        hypotheses = phd_hyp.hypothesise(current_state,yk,times[k],
                                         order_by_detection=True)
        updated_states = phd_up.update(hypotheses)
        reduced_states = set(reducer.reduce(updated_states))
        for reduced_state in reduced_states:
            tag = reduced_state.tag
            if reduced_state.weight > state_threshold:
                for track in phd_tracks:
                    track_tags = [state.tag for state in track.states]
                    if tag in track_tags:
                        track.append(reduced_state)
                        break
                else:
                    new_track = Track(reduced_state)
                    phd_tracks.add(new_track)
   
    # save PHD results
    if saving_results:
        name = f'Results/phd{set_num}.npz'
        all_metrics[set_ind,:] = f.saveSS(name,phd_tracks,metric_quantities)
        if wanting_main_plot:
            MQ_results = metric_quantities[:2] + metric_quantities[3:]
            tracks,births,_ = f.open_results('GM-PHD',set_num,MQ_results)
            deaths = np.array([births[i] + len(tracks[i])
                               for i in range(len(tracks))],int)
            f.plot_results(y,truths,true_deaths,tracks,deaths,S,colours,
                           'GM-PHD',aspect)
    elif wanting_main_plot:
        print('must save results; change file name if overwriting undesired')