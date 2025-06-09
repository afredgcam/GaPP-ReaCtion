# GNN-CV

import numpy as np
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.models.transition.linear import \
    CombinedLinearGaussianTransitionModel as CLGTM, ConstantVelocity as CV
from stonesoup.types.detection import Detection
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.state import GaussianState
from datetime import datetime,timedelta
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment as GNN
from stonesoup.initiator.simple import MultiMeasurementInitiator as MMI
from stonesoup.deleter.error import CovarianceBasedDeleter
from stonesoup.deleter.time import UpdateTimeStepsDeleter as UTSD
from stonesoup.deleter.multi import CompositeDeleter
from tqdm import tqdm
import matplotlib.pyplot as plt
import functions as f

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
    s2_prior = 1
    
    # other parameters
    Q = 100 # CV driving noise variance
    max_obs_dist = 3 # cannot associate data farther than this (Mahalanobis)
    zero_count_limit = 3 # from gapp_reaction_pooled.py
    sd_lim = 10
    min_track_pts = 5
    
    # convert data for stone soup
    measurement_model = LinearGaussian(4,(0,2),s2_prior*np.eye(2))
    all_measurements = [{Detection(y[k][l:l+1,:].T,times[k],measurement_model)
                         for l in range(len(y[k]))} for k in range(Tmax)]

    # CV transition model
    transition_model = CLGTM([CV(Q),CV(Q)])
    
    # Kalman filter
    pred = KalmanPredictor(transition_model)
    upd = KalmanUpdater(measurement_model)
    
    # data association
    hypothesiser = DistanceHypothesiser(pred,upd,measure=Mahalanobis(),
                                        missed_distance=max_obs_dist)
    data_associator = GNN(hypothesiser)
    
    # deleter
    cov_deleter = CovarianceBasedDeleter(sd_lim**2,mapping=[0,2])
    step_deleter = UTSD(zero_count_limit)
    deleter = CompositeDeleter([cov_deleter,step_deleter],intersect=False)
    
    # initiator
    prior_var = np.diag([s2_prior,100,s2_prior,100]) # vel. var from gm_phd.py
    prior = GaussianState(np.zeros([4,1]),prior_var)
    initiator = MMI(prior,deleter,data_associator,upd,measurement_model,
                    min_track_pts)   
    
    # run tracker
    tracks,all_tracks = set(),set()
    for k in time_steps:
        # data at k
        obs = all_measurements[k]
        # hypothesise
        hypotheses = data_associator.associate(tracks,obs,times[k])
        # update by hypotheses
        associated_obs = set()
        for track in tracks:
            hypothesis = hypotheses[track]
            if hypothesis.measurement:
                post = upd.update(hypothesis)
                track.append(post)
                associated_obs.add(hypothesis.measurement)
            else:
                # take prediction if no measurement suitable
                track.append(hypothesis.prediction)
        # track management
        tracks -= deleter.delete_tracks(tracks)
        other_obs = obs - associated_obs
        tracks |= initiator.initiate(other_obs,times[k])
        all_tracks |= tracks
    
    # save results
    if saving_results:
        name = f'Results/gnn{set_num}.npz'
        all_metrics[set_ind,:] = f.saveSS(name,all_tracks,metric_quantities)
        if wanting_main_plot:
            MQ_results = metric_quantities[:2] + metric_quantities[3:]
            tracks,births,_ = f.open_results('GNN-CV',set_num,MQ_results)
            deaths = np.array([births[i] + len(tracks[i])
                               for i in range(len(tracks))],int)
            f.plot_results(y,truths,true_deaths,tracks,deaths,S,colours,
                           'GNN-CV',aspect)
    elif wanting_main_plot:
        print('must save results; change file name if overwriting undesired')
