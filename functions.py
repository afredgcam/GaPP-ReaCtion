# GaPP-Class and GaPP-ReaCtion functions

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import norm
from scipy.linalg import solve,det
from scipy.optimize import minimize,linear_sum_assignment
from copy import deepcopy as dc
import warnings
import matplotlib.pyplot as plt
from stonesoup.types.track import Track
from stonesoup.types.groundtruth import GroundTruthState,GroundTruthPath
from stonesoup.types.state import State
from datetime import datetime,timedelta
from stonesoup.dataassociator.tracktotrack import TrackToTruth


# get global random number generator
rng = np.random.default_rng(seed=28)





##### classes for data generation #####

# truth track
class Truth_iSE:
    
    # start track at X0
    def __init__(self,k,Tmax,X0,parameters,dims=2):
        d,σ2,ell,μ,class_num = parameters
        self.d = d # iSE independence time (step) difference
        self.σ2 = σ2 # iSE scale hyperparameter
        self.ell = ell # iSE length-scale hyperparameter
        self.μ = μ # E(detections / time step)
        self.class_num = class_num # class label
        
        # initial location
        self.X = np.zeros([1,dims])
        self.X[0,:] = X0.reshape(-1)
        # record birth and (prospective) death times
        self.κ_birth = k
        self.κ_death = Tmax
        # initial activity (active = True = alive, inactive = False = dead)
        self.ζ = [True]
        # prepare for observations
        self.data = []
    
    # survival
    def survival(self,ψ):
        if not self.ζ[-1]:
            # dead stay dead
            self.ζ.append(False)
        else:
            # activity sampled (never dies in first d steps)
            self.ζ.append(rng.random()*(self.X.shape[0] > self.d) < ψ)
            if not self.ζ[-1]:
                # record time of death
                self.κ_death = self.κ_birth + self.X.shape[0]
    
    # move track
    def move(self,dt):
        if not self.ζ[-1]:
            # skip if track inactive
            return None
        if self.X.shape[0] < self.d:
            # state dimension grows for first d-1 steps
            t = dt * np.arange(self.X.shape[0]+1)
            C = iSE(t,t,self.σ2,self.ell) + 1e-5 * dt**2
            # get iSE positional distribution
            gtw = solve(C[:-1,:-1],C[:-1,-1:],assume_a='pos')
            Xk_mean = (self.X[0,:] + gtw.T @ (self.X - self.X[0,:]))[0,:]
            Xk_var = (C[-1:,-1:] - C[-1:,:-1] @ gtw)[0,0]
            # sample and record new position
            Xk = Xk_mean + rng.normal(0,1,size=self.X.shape[1]) * Xk_var**0.5
            self.X = np.vstack([self.X,Xk.reshape([1,-1])])
        else:
            # state retains length d
            t = dt * np.arange(self.d) + dt
            C = iSE(t,t,self.σ2,self.ell)
            # get iSE positional distribution
            g = solve(C[:-1,:-1],C[:-1,-1:],assume_a='pos')
            G = np.zeros([1,self.d])
            G[:1,1:] = g.T
            G[0,0] = 1 - g.sum()
            Xk_mean = G @ self.X[-self.d:,:]
            Xk_var = (C[-1:,-1:] - C[-1:,:-1] @ g)[0,0]
            # sample and record new position
            Xk = Xk_mean + rng.normal(0,1,size=self.X.shape[1]) * Xk_var**0.5
            self.X = np.vstack([self.X,Xk.reshape([1,-1])])
    
    # get observations
    def be_observed(self,s2):
        if not self.ζ[-1]:
            # skip if track inactive
            return None
        # number of observations
        ni = rng.poisson(self.µ)
        # current position
        Xk = self.X[-1,:]
        # generate observations
        data_k = Xk + rng.normal(0,1,size=[ni,2]) * s2**0.5
        # add them to the data set
        self.data.append(data_k)
    
    # plot track (at given steps – untested feature!)
    def plot_track(self,colour,style,mrks,name,steps=None):
        if type(steps) == type(None):
            # plot whole track
            end_mk = mrks[2-self.ζ[-1]] # different marker if survived or not
            plt.plot(self.X[:,0],self.X[:,1],c=colour,ls=style,label=name)
            plt.plot(self.X[0,0],self.X[0,1],c=colour,ls='',marker=mrks[0])
            plt.plot(self.X[-1,0],self.X[-1,1],c=colour,ls='',marker=end_mk)
        else:
            # plot track section
            if self.κ_death <= steps[0] or steps[-1] < self.κ_birth:
                # not active between these times, so plot nothing
                return None
            # find where these steps sit in the life of the track
            max_ind = min(steps[-1]+1,self.κ_death)
            inds = np.arange(max(self.κ_birth,steps[0]),max_ind) - self.κ_birth
            end_mk = mrks[2] if steps[-1]-self.κ_death >= 0 else mrks[1]     
            plt.plot(self.X[inds,0],self.X[inds,1],c=colour,ls=style,
                     label=name)
            plt.plot(self.X[inds[0],0],self.X[inds[0],1],c=colour,ls='',
                     marker=mrks[0])
            plt.plot(self.X[inds[-1],0],self.X[inds[-1],1],c=colour,ls='',
                     marker=end_mk)


# data set class
class DataSet:
    
    # initialise data set
    def __init__(self,Tmax,dt,d,s2_range,ψ,γ_range,µ0_range,Nc,class_pars,
                 µi_range,dims=2,padding=0.2):
        self.Tmax = Tmax # total time steps
        self.dt = dt # time step size
        self.d = d # iSE independence time (step) difference
        s2_min,s2_Max = s2_range
        self.s2 = s2_min + (s2_Max - s2_min) * rng.random() # observation variance
        self.ψ = ψ # survival probability
        γ_min,γ_Max = γ_range
        self.γ = γ_min + (γ_Max - γ_min) * rng.random() # E(no. new objects / step)
        µ0_min,µ0_Max = µ0_range # note: this may be an intensity range
        self.µ0 = µ0_min + (µ0_Max - µ0_min) * rng.random() # clutter rate / intensity
        self.Nc = Nc # number of object classes
        self.class_pars = class_pars # iSE hyperparameters for each class
        self.µi_range = µi_range # min / max object detection rates
        self.dims = dims # dimensionality (i.e., dims=2 => (x,y) coörds only)
        self.padding = padding # space between edge of most distant track and edge of scene
        
        self.η = np.zeros(0,int) # number of new objects at each step
        self.objects = [] # tracks etc.
    
    # generate object
    def generate_object(self,k):
        # sample class
        Ci = rng.choice(self.Nc)
        σ2i,ell_i = self.class_pars[Ci,:]
        # sample detection rate
        µ_min,µ_Max = self.µi_range
        µi = µ_min + (µ_Max - µ_min) * rng.random()
        # record and save hyperparameters
        parameters = [self.d,σ2i,ell_i,µi,Ci]
        # make track
        new_obj = Truth_iSE(k,self.Tmax,np.zeros(self.dims),parameters)
        self.objects.append(new_obj)
    
    # ensure starts with ≥1 objects
    def first_step(self):
        # generate any new objects
        η = rng.poisson(self.γ) + 1
        self.η = np.append(self.η,η)
        for _ in range(η):
            self.generate_object(0)
        # observe objects
        for truth in self.objects:
            truth.be_observed(self.s2)
    
    # generate next step from generative model
    def random_next_step(self,k):
        # cull and move objects
        for truth in self.objects:
            truth.survival(self.ψ)
            truth.move(self.dt) 
        # generate any new objects
        η = rng.poisson(self.γ)
        self.η = np.append(self.η,η)
        for _ in range(η):
            self.generate_object(k)
        # observe objects
        for truth in self.objects:
            truth.be_observed(self.s2)
    
    # no new objects near end
    def late_step(self,k):
        # cull and move objects
        for truth in self.objects:
            truth.survival(self.ψ)
            truth.move(self.dt)   
        # observe objects
        for truth in self.objects:
            truth.be_observed(self.s2)
    
    # shift objects (and their data) to be more central
    def shift_objects(self,sparse):
        # centralise objects
        for target in self.objects:
            X_mins,X_maxs = target.X.min(0),target.X.max(0)
            dists = X_maxs - X_mins
            target.X -= X_mins + dists/2
            target.data = [y - X_mins - dists/2 for y in target.data]
        # make 2d scene
        all_max = [max([target.X.max(0)[0] for target in self.objects]),
                   max([target.X.max(0)[1] for target in self.objects])]
        S = np.array([all_max]).T @ np.array([[-1,1]]) * (1+self.padding)
        if sparse:
            # spread scene
            S *= len(self.objects) ** 0.5
            # enlarge to square
            S_max = S.max()
            S[:,0],S[:,1] = -S_max,S_max
        # scatter objects around scene
        for target in self.objects:
            X_maxs = target.X.max(0)
            max_shift = S[:,1] - X_maxs
            shifts = rng.choice([-1,1],size=2) * rng.random(2) * max_shift
            target.X += shifts
            target.data = [y + shifts for y in target.data]
        # finalise scene
        S *= 1 + self.padding
        self.S = S
        # save aspect ratio
        nat_aspect = S[0,1] / S[1,1]
        self.aspect = 5 * np.array([nat_aspect,1])
    
    # make clutter
    def add_clutter(self,rate_given):
        if not rate_given:
            # get rate from intensity
            self.µ0 *= (self.S[:,1] - self.S[:,0]).prod()
        self.clutter = []
        for _ in range(self.Tmax):
            # number of clutter
            n0 = rng.poisson(self.µ0)
            # location of clutter
            y0 = (2*rng.random(size=[n0,self.dims]) - 1) * self.S[:,1]
            self.clutter.append(y0)
    
    # generate full data set
    def generate(self,sparse=False,rate_given=True):
        # make trajectories and their data
        for k in range(self.Tmax):
            if k == 0:
                self.first_step()
            elif k < self.Tmax - self.d:
                self.random_next_step(k)
            else:
                self.late_step(k)
        # shift objects and add clutter
        self.shift_objects(sparse)
        self.add_clutter(rate_given)
        # combine data
        self.y,self.a = [],[]
        for k in range(self.Tmax):
            # prepare obs and associations at step k
            yk = np.zeros([0,self.dims])
            ak = np.zeros(0,int)
            for i in range(len(self.objects)):
                target = self.objects[i]
                if not target.κ_birth <= k < target.κ_death:
                    # no observations if inactive at k, so skip
                    continue
                # add data for object i at step k
                yki = target.data[k-target.κ_birth]
                yk = np.vstack([yk,yki])
                # add corresponding associations
                ak = np.append(ak,np.ones(yki.shape[0],int)*(i+1))
            # add clutter at step k
            yk0 = self.clutter[k]
            yk = np.vstack([yk,yk0])
            # add clutter associations
            ak = np.append(ak,np.zeros(yk0.shape[0],int))
            # add data / associations at step k to all data / associations lists
            self.y.append(yk)
            self.a.append(ak)
    
    # save data
    def save_data(self,name):
        parameters = np.array([self.Tmax,self.dt,self.d,self.s2,self.ψ,self.γ,
                               self.µ0,self.Nc,self.class_pars,self.µi_range],
                              object)
        tracks = np.array([target.X for target in self.objects],object)
        Po_means = np.array([target.µ for target in self.objects])
        classes = np.array([target.class_num for target in self.objects],int)
        births = np.array([target.κ_birth for target in self.objects],int)
        deaths = np.array([target.κ_death for target in self.objects],int)
        status = np.array([target.ζ for target in self.objects],object)
        yis = np.array([target.data for target in self.objects],object)
        np.savez(name,data=np.array(self.y,object),status=status,yis=yis,
                 associations=np.array(self.a,object),deaths=deaths,
                 tracks=tracks,Po_means=Po_means,classes=classes,births=births,
                 scene=self.S,aspect=self.aspect,parameters=parameters)     
    
    # plot data set
    def plot(self,colours,style,mrks):
        # set aspect ratio
        plt.rcParams["figure.figsize"] = self.aspect
        name = 'Obs.'
        for yk in self.y:
            plt.plot(yk[:,0],yk[:,1],'k2',alpha=0.05,label=name)
            name = ''
        name = 'Truth'
        for i in range(len(self.objects)):
            target = self.objects[i]
            colour = colours[i%len(colours)]
            target.plot_track(colour,style,mrks,name)
            name = ''
        plt.xlim(self.S[0,:])
        plt.ylim(self.S[1,:])
        plt.legend(loc=0,framealpha=1)
        plt.show()
    
    # plot each time step – untested
    def plot_step(self,k,colours,style,mrks):
        # set aspect ratio
        plt.rcParams["figure.figsize"] = self.aspect
        yk = self.y[k]
        plt.plot(yk[:,0],yk[:,1],'k2',alpha=0.3,label='Obs.')
        steps = np.arange(k-self.d,k) + 1
        for i in range(len(self.objects)):
            target = self.objects[i]
            colour = colours[i%len(colours)]
            target.plot_track(colour,style,mrks,'',steps)
        plt.plot(self.S[0,0]-10*np.ones(2),[0,1],c='blueviolet',ls=style,
                 marker='',label='Truth')
        plt.xlim(self.S[0,:])
        plt.ylim(self.S[1,:])
        plt.legend(loc=0,framealpha=1)
        plt.show()

#######################################



##### class for iSE track #####

class Track_iSE:
    
    # initialise iSE track
    def __init__(self,yi,s2,k,props):
        
        # unpack properties
        d,dζ,dt,Nc,hPars,αp,βp,rate,zero_lim,sd_lim = props
        
        # default parameters for all tracks
        self.d = d
        self.dζ = dζ
        self.dt = dt
        self.num_classes = Nc
        self.hyperpars = hPars
        self.α_plus = αp
        self.β_plus = βp
        
        # default deletion criteria
        self.rate_limit = rate
        self.zero_count_limit = zero_lim
        self.sd_limit = sd_lim
        # start properties
        self.id = rng.random()
        self.κ_birth = k
        self.ζ = [True]
        self.Ω = [False]
        self.ζ_expo = 0
        
        # initial state properties
        X0 = yi.mean(0)
        n = yi.shape[0]
        V0 = s2 / n
        self.m_preds = [np.array([X0]) for _ in range(Nc)] # mean matrices of prediction ∀ classes
        self.V_preds = [np.array([[V0]]) for _ in range(Nc)] # variance matrix of prediction ∀ classes
        self.up_m = [np.array([X0]) for _ in range(Nc)] # mean matrix of update ∀ classes
        self.up_V = [np.array([[V0]]) for _ in range(Nc)] # variance matrix of update ∀ classes
        self.up_π = np.ones(self.num_classes) / self.num_classes # current class probabilities
        # updated µi hyperparameters
        self.α = self.α_plus + n
        self.β = self.β_plus + 1
        # record keeping
        self.X = np.array([X0]) # real-time location means
        self.Var = V0 * np.array([np.eye(X0.shape[0])]) # real-time location variances
        self.X_best = np.zeros([0,2]) # fixed-lag smoothing location means
        self.Var_best = np.zeros([0,2,2]) # FLS location variances
        self.π = np.ones([1,self.num_classes]) / self.num_classes # real-time class probabilities
        # real-time µi hyperparameters
        self.α_all = np.array([αp,self.α])
        self.β_all = np.array([βp,self.β])
        self.n = np.array([n]) # number of sampled associations

    
    # survival
    def survival(self,ψ,S):
        if not self.ζ[-1]:
            # deleted tracks stay dead (possibly allowing revival)
            self.ζ.append(False)
            k_κp = self.get_k_minus_κ_prime()
            self.Ω.append(bool(self.Ω[-1] * (sum(self.Ω)<self.dζ-k_κp)))
            self.ζ_expo = 0
            return None
        # active tracks consider deletion
        Nc = self.num_classes
        di,dim = self.up_m[0].shape
        up_m0 = [self.up_m[c][0,:] for c in range(Nc)]
        up_V0 = [self.up_V[c][0,0] * np.eye(dim) for c in range(Nc)]
        # find Gaussian approximation to Gaussian mixture
        m_avg,V_avg = collapse(up_m0,up_V0,self.up_π)
        # is E(µi) too low?
        low_rate = self.α/self.β < self.rate_limit
        # has this object not associated any observations for too long?
        missed_detection = self.n[-self.zero_count_limit:].sum() == 0
        # is the object still present in the scene?
        in_scene = bool(np.prod([S[dim,0]<m_avg[dim]<S[dim,1]
                                 for dim in range(S.shape[0])]))
        # is the location variance too high?
        high_var = det(V_avg)**0.5 > self.sd_limit
        # check if deleting by these heuristics
        deleting = low_rate or missed_detection or not in_scene or high_var
        if deleting:
            # heuristic deletion
            self.ζ.append(False)
            self.Ω.append(False)
            self.ζ_expo = 1
        else:
            # sampled survival / deletion
            self.ζ.append(rng.random() < ψ)
            self.Ω.append(not self.ζ[-1])
            self.ζ_expo = 1 - self.ζ[-1]
        if not self.ζ[-1]:
            self.certify_death(m_avg,V_avg)
                
    
    # death certificate (update records upon death)
    def certify_death(self,m_avg,V_avg):
        Nc = self.num_classes
        di,dim = self.up_m[0].shape
        self.κ_death = self.κ_birth + self.X.shape[0]
        if self.X_best.shape[0] != 0:
            best_Xs,best_Vs = [m_avg],[V_avg]
            for k in range(1,self.d-1):
                up_mk = [self.up_m[c][k,:] for c in range(Nc)]
                up_Vk = [self.up_V[c][k,k] * np.eye(dim) for c in range(Nc)]
                m_avg,V_avg = collapse(up_mk,up_Vk,self.up_π)
                best_Xs.append(m_avg)
                best_Vs.append(V_avg)
            best_Xs,best_Vs = np.array(best_Xs[::-1]),np.array(best_Vs[::-1])
            self.X_best = np.vstack([self.X_best,best_Xs])
            self.Var_best = np.vstack([self.Var_best,best_Vs])
        else:
            best_Xs,best_Vs = [m_avg],[V_avg]
            for k in range(1,di):
                up_mk = [self.up_m[c][k,:] for c in range(Nc)]
                up_Vk = [self.up_V[c][k,k] * np.eye(dim) for c in range(Nc)]
                m_avg,V_avg = collapse(up_mk,up_Vk,self.up_π)
                best_Xs.append(m_avg)
                best_Vs.append(V_avg)
            best_Xs,best_Vs = np.array(best_Xs[::-1]),np.array(best_Vs[::-1])
            self.X_best = np.vstack([self.X_best,best_Xs])
            self.Var_best = np.vstack([self.Var_best,best_Vs])
    
    
    # overturn death (keeps length of time the same)
    def overturn_death(self):
        self.ζ = [True] * len(self.ζ)
        self.Ω = [False] * len(self.ζ)
        self.κ_death = None
        self.X_best = self.X_best[:max(len(self.ζ)-self.d,0),:]
        self.Var_best = self.Var_best[:max(len(self.ζ)-self.d,0)]
    
    
    # predict
    def predict(self):
        if self.ζ[-1] + self.Ω[-1] == 0:
            # skip predict if inactive or can't be revived
            return None
        self.m_preds,self.V_preds = [],[]
        k = self.up_m[0].shape[0]
        if k < self.d:
            # state dimension grows
            t = self.dt * np.arange(k,-1,-1) 
            for c in range(self.num_classes):
                # compute mean and variance matrices
                [σ2,ell] = self.hyperpars[c,:]
                C = iSE(t,t,σ2,ell) + self.Var[0][0,0]
                gtw = solve(C[1:,1:],C[1:,:1],assume_a='pos')
                Gtw = np.eye(k+1,k,-1)
                Gtw[:1,:] = gtw.T
                qch = C[:1,:1] - C[:1,1:] @ gtw
                Qtw = np.zeros([k+1,k+1])
                Qtw[:1,:1] = qch
                mcch = Gtw @ (self.up_m[c] - self.X[0,:]) + self.X[0,:]
                # save them as current predict step
                self.m_preds.append(mcch)
                self.V_preds.append(Gtw @ self.up_V[c] @ Gtw.T + Qtw)
        else:
            # state retains length d
            t = self.dt * np.arange(self.d,0,-1)
            for c in range(self.num_classes):
                # compute mean and variance matrices
                [σ2,ell] = self.hyperpars[c,:]
                C = iSE(t,t,σ2,ell)
                g = solve(C[1:,1:],C[1:,:1],assume_a='pos')
                G = np.eye(self.d,k=-1)
                G[:1,:-1] = g.T
                G[0,-1] = 1 - g.sum()
                qhat = C[:1,:1] - C[:1,1:] @ g
                Q = np.zeros([self.d,self.d])
                Q[:1,:1] = qhat
                # save them as current predict step
                self.m_preds.append(G @ self.up_m[c])
                self.V_preds.append(G @ self.up_V[c] @ G.T + Q)
    
    
    # update
    def update(self,yi,s2):
        
        # useful quantities
        n,dims = yi.shape
        Nc = self.num_classes
        
        # position parameters for each class
        up_mc,up_Vc = dc(self.m_preds),dc(self.V_preds)
        log_class_weights = np.zeros(Nc)
        if n != 0:
            y_bar = yi.mean(0).reshape([1,-1])
            for c in range(self.num_classes):
                mcch,Vcch = self.m_preds[c],self.V_preds[c]
                factor = n / (n*Vcch[0,0]+s2)
                up_mc[c] = mcch + factor * Vcch[:,:1] @ (y_bar-mcch[:1,:])
                up_Vc[c] = Vcch - factor * Vcch[:,:1] @ Vcch[:1,:]
                mc1,Vc11 = self.m_preds[c][0,:],self.V_preds[c][0,0]
                log_lik_c = log_lik_offset_var(yi,mc1,Vc11,s2)
                log_class_weights[c] = log_lik_c + np.log(self.up_π[c]+1e-10)
            # class probabilities
            self.up_π = normalise(log_class_weights)
        # save positional update
        self.up_m = up_mc
        self.up_V = up_Vc
        
        # skip rest if inactive
        if not self.ζ[-1]:
            return None
        
        # count hyperparameters
        self.α += n
        self.β += 1
        
        # keep records tidy
        up_m0 = [up_mc[c][0,:] for c in range(Nc)]
        up_V0 = [up_Vc[c][0,0]*np.eye(dims) for c in range(Nc)]
        m0_avg,V0_avg = collapse(up_m0,up_V0,self.up_π)
        m0_avg,V0_avg = np.array([m0_avg]),np.array([V0_avg])
        self.X = np.vstack([self.X,m0_avg])
        self.Var = np.vstack([self.Var,V0_avg])
        if self.up_m[0].shape[0] == self.d:
            up_md = [up_mc[c][-1,:] for c in range(Nc)]
            up_Vd = [up_Vc[c][-1,-1]*np.eye(dims) for c in range(Nc)]
            md_avg,Vd_avg = collapse(up_md,up_Vd,self.up_π)
            md_avg,Vd_avg = np.array([md_avg]),np.array([Vd_avg])
            self.X_best = np.vstack([self.X_best,md_avg])
            self.Var_best = np.vstack([self.Var_best,Vd_avg])
        self.π = np.vstack([self.π,self.up_π.reshape([1,-1])])
        self.α_all = np.append(self.α_all,self.α)
        self.β_all = np.append(self.β_all,self.β)
        self.n = np.append(self.n,n)
    
    
    # undo update
    def downdate(self,yi,s2):
        
        # skip if inactive
        if not self.ζ[-1]:
            return None
        
        # useful quantities
        n = yi.shape[0]
        
        # remove latest class probabilities
        self.π = self.π[:-1,:]
        self.up_π = self.π[-1,:]
        
        # count hyperparameters
        self.α -= n
        self.β -= 1
        
        # keep records tidy
        self.X = self.X[:-1,:]
        self.Var = self.Var[:-1,:,:]
        if self.m_preds[0].shape[0] == self.d:
            self.X_best = self.X_best[:-1,:]
            self.Var_best = self.Var_best[:-1,:,:]
        self.α_all = self.α_all[:-1]
        self.β_all = self.β_all[:-1]
        self.n = self.n[:-1]
    
    
    # find k - κ'
    def get_k_minus_κ_prime(self,skip_last=False):
        n_vec = self.n[:self.n.shape[0]-int(skip_last)]
        k_κp = len(self.ζ) - np.where(n_vec>0)[0][-1] - 1
        return k_κp
    
    
    # add to plot
    def plot(self,colour='blueviolet',line='-',markers=['x','.'],name='Track',
             α=1):
        plt.plot(self.X[:,0],self.X[:,1],c=colour,ls=line,label=name,alpha=α)
        for i in [0,-1]:
            plt.plot(self.X[i,0],self.X[i,1],c=colour,ls='',marker=markers[i],
                     label='',alpha=α)
    
###############################



##### class for individual particles #####


class Particle:
    
    def __init__(self,J,particle_props,track_props,scene_props,j):
        
        # unpack properties
        ψ,α0_plus,β0_plus,ε_plus,ξ_plus,φ_plus,b_plus = particle_props
        ρ,S = scene_props[:-1]
        
        # id
        self.id = j
        
        # global hyperparameters
        self.w = 1/J # particle weight
        self.ψ = ψ # survival probability
        # clutter rate hyperparameters
        self.α0 = α0_plus
        self.β0 = β0_plus
        # object birth count hyperparameters
        self.ε = ε_plus
        self.ξ = ξ_plus
        # observation variance hyperparameters
        self.φ = φ_plus
        self.b = b_plus
        
        # tracks
        self.tracks = []
        self.track_props = track_props
        
        # scene
        self.ρ = ρ # inverse of scene's area
        self.S = S # scene corner coördinates
        
        # record keeping
        self.η = np.zeros(0,int) # number of births / time step
        self.a_all = [] # data associations sampled
        self.α0_all = np.array([α0_plus])
        self.β0_all = np.array([β0_plus])
        self.ε_all = np.array([ε_plus])
        self.ξ_all = np.array([ξ_plus])
        self.φ_all = np.array([φ_plus])
        self.b_all = np.array([b_plus])
    
    
    # survival deletion step
    def survival(self):
        for track in self.tracks:
            track.survival(self.ψ,self.S)
    
    
    # predict step
    def predict(self):
        for track in self.tracks:
            track.predict()
        
    
    # data association
    def associate(self,y,G):
        
        n,dim = y.shape
        if n == 0:
            # handle 'no data' case
            self.log_Qch = 0
            self.a = np.zeros(0,int)
            self.a_all.append(dc(self.a))
            N_new = 0
            self.η = np.append(self.η,N_new)
            return None
            
        
        # useful quantities
        Θ = len(G)
        N = len(self.tracks)
        Nc,_,αp,βp = self.track_props[3:7]
        s2 = self.b / (self.φ - 1)
        
        # current predicted means / variances / weights
        ms = np.array([[x.m_preds[c][0,:] for c in range(Nc)]
                       for x in self.tracks]) # (N x Nc x dims)
        Vs = np.array([[x.V_preds[c][0,0] for c in range(Nc)]
                       for x in self.tracks]) # (N x Nc)
        πs = np.array([x.up_π for x in self.tracks]) # (N x Nc)
        
        # unnormalised initial association sampling distributions
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        log_priors,log_liks = np.zeros([Θ,N+2]),np.zeros([Θ,N+2])
        log_priors[:,0] = [np.log(self.α0/self.β0) * len(Gθ) for Gθ in G]
        log_priors[:,1:-1] = np.array([[np.log(x.ζ[-1])+len(Gθ)*np.log(x.α/x.β)
                                        for x in self.tracks] for Gθ in G])
        log_priors[:,-1] = [(len(Gθ)*np.log(αp/βp)*np.log(int(len(Gθ)>1)))
                            for Gθ in G]
        log_priors[:,-1] += np.log(1 - (self.ξ/(1+self.ξ))**self.ε)
        log_liks[:,0] = [len(Gθ) * np.log(self.ρ) for Gθ in G]
        if N > 0:
            likc = np.array([[[np.exp(log_lik_offset_var(y[Gθ,:],ms[i,c,:],
             Vs[i,c],s2)) for c in range(Nc)] for i in range(N)] for Gθ in G]) # (Θ x N x Nc)
            log_liks[:,1:-1] = np.log(np.einsum('ijk,jk->ij',likc,πs))
        log_liks[:,-1] = [dim*np.log(2*np.pi*s2/len(Gθ))/2 for Gθ in G]
        log_liks[:,-1] += [norm.logpdf(y[Gθ,:],y[Gθ,:].mean(0),s2**0.5).sum()
                           for Gθ in G]
        log_liks[:,-1] += np.log(self.ρ)
        unnorm_qAch = log_priors + log_liks
        unnorm_qAch[np.isinf(unnorm_qAch)] = -1e+200
        warnings.filterwarnings("default", category=RuntimeWarning)
        
        # sample initial associations
        qAch = np.array([normalise(unnorm_qAch[θ,:]) for θ in range(Θ)])
        Ach = np.array([rng.choice(N+2,p=qAch[θ,:])for θ in range(Θ)],int)
        self.log_Qch = np.log(qAch[np.arange(Θ),Ach]).sum()
        
        # sample final associations
        N_new = 0
        new_y_bar,new_n = [],[]
        A = np.zeros(Ach.shape,int)
        for θ in range(Θ):
            if Ach[θ] != N+1:
                A[θ] = Ach[θ]
                continue
            nθ,yθ_bar = len(G[θ]),y[G[θ],:].mean(0)
            log_qA = np.zeros(N_new+1)
            log_qA[:-1] = [dim*np.log(nθ*nι/(nθ+nι))/2 for nι in new_n]
            log_qA[:-1] -= [nθ*nι/(nθ+nι) * ((yθ_bar-y_ι)**2).sum() / (2*s2)
                            for nι,y_ι in zip(new_n,new_y_bar)]
            log_qA[-1] = np.log(self.ρ) + dim * np.log(2*np.pi*s2)/2 
            log_qA[-1] += np.log(self.ε + N_new) - np.log(self.ξ + 1)
            qA = normalise(log_qA)
            A[θ] = rng.choice(N_new+1,p=qA) + N + 1
            self.log_Qch += np.log(qA[A[θ]-N-1])
            if A[θ] == N_new + N + 1:
                new_n.append(nθ)
                new_y_bar.append(yθ_bar)
                N_new += 1
            else:
                nAyA = new_n[A[θ]-N-1] * new_y_bar[A[θ]-N-1]
                new_nA = new_n[A[θ]-N-1] + nθ
                new_y_bar[A[θ]-N-1] = (nAyA + nθ * yθ_bar) / new_nA
                new_n[A[θ]-N-1] = new_nA
        
        # synthesise associations
        self.a = np.zeros(n,int)
        for θ in range(Θ):
            self.a[G[θ]] = A[θ]
        
        # keep records tidy
        self.a_all.append(dc(self.a))
        self.η = np.append(self.η,N_new)
    
    
    # update particle (inc. all tracks)
    def update(self,y,k):
        
        # useful quantities
        N = len(self.tracks)
        
        # update hyperparameters (except s2)
        self.α0 += (self.a == 0).sum()
        self.β0 += 1
        self.ε += self.η[-1]
        self.ξ += 1
        
        # update s2
        for i in range(N+self.η[-1]):
            yi = y[self.a==i+1,:]
            ni = yi.shape[0]
            if ni < 2:
                # skip if no variance visible
                continue
            yibar = yi.mean(0)
            self.b += ((yi - yibar)**2).sum() / 2
            self.φ += (ni-1) / 2
        s2 = self.b / (self.φ - 1)
        
        # update pre-existing tracks
        for i in range(N):
            yi = y[self.a==i+1,:]
            self.tracks[i].update(yi,s2)
        
        # initialise new tracks
        for i in range(N,N+self.η[-1]):
            yi = y[self.a==i+1,:]
            new_track = Track_iSE(yi,s2,k,self.track_props)
            self.tracks.append(new_track)
        
        # keep records tidy
        self.α0_all = np.append(self.α0_all,self.α0)
        self.β0_all = np.append(self.β0_all,self.β0)
        self.ε_all = np.append(self.ε_all,self.ε)
        self.ξ_all = np.append(self.ξ_all,self.ξ)
        self.φ_all = np.append(self.φ_all,self.φ)
        self.b_all = np.append(self.b_all,self.b)
    
    
    # pass through revival / split kernel
    def revival(self,y,k):
        
        # useful quantities
        Nk = len(self.tracks)
        ηk = self.η[-1]
        Nk_1 = Nk - ηk
        log_ρ = np.log(self.ρ)
        log_ψ = np.log(self.ψ)
        log_1_ψ = np.log(1 - self.ψ)
        εk_1 = self.ε_all[-2]
        ξk_1 = self.ξ_all[-2]
        d = self.track_props[0]
        αp = self.track_props[5]
        βp = self.track_props[6]
        Nc = self.track_props[3]
        s2 = self.b / (self.φ - 1)
        dims = self.S.shape[0]
        
        # revival counter / record
        R = 0
        revived = []
        
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        # revivals
        for ι in range(Nk_1,Nk):
            
            # prepare log proposal density
            log_qr = np.zeros(Nk_1+1)
            
            # ι quantities
            yι = y[self.a==ι+1-R,:]
            yιbar = yι.mean(0)
            nι = yι.shape[0]
            Lι = log_ρ + dims * np.log(2*np.pi*s2 / nι) / 2
            Lι += norm.logpdf(yι,yιbar,s2**0.5).sum()
            
            for i in range(Nk_1):
                xi = self.tracks[i]
                if xi.Ω[-1] == 0:
                    # skip if ineligible for revival
                    log_qr[i+1] = -np.inf
                    continue
                # compute log(qr(i|ι))
                L_iι = np.array([log_lik_offset_var(yι,xi.m_preds[c][0,:],
                                 xi.V_preds[c][0,0],s2) for c in range(Nc)])
                L_marg_iι = np.log((np.exp(L_iι)*xi.up_π).sum())
                ψ_term = (k - xi.κ_death + 1) * log_ψ - log_1_ψ
                µ_term = log_gamma_ratio(xi.α,nι) + xi.α * np.log(xi.β)
                µ_term -= (xi.α + nι) * np.log(xi.β + k - xi.κ_death + 1)
                µ_term -= log_gamma_ratio(αp,nι) + αp * np.log(βp)
                µ_term += (αp + nι) * np.log(βp + 1)
                γ_term = np.log(ηk-R) + np.log(ξk_1+1) - np.log(εk_1+ηk-R-1)
                log_qr[i+1] = L_marg_iι - Lι + ψ_term + µ_term + γ_term
            
            # propose revival
            qr = normalise(log_qr)
            I = rng.choice(Nk_1+1,p=qr)
            if I > 0:
                # sample acceptance
                xI = self.tracks[I-1]
                k_κp = xI.get_k_minus_κ_prime()
                λιI = np.exp(log_qr).sum() / k_κp
                if rng.random() < λιI:
                    # accept proposal: amend particle
                    R += 1
                    revived.append(I-1)
                    time_dead = len(xI.ζ) - sum(xI.ζ)
                    # update old records
                    if time_dead > 1:
                        xI.α_all = np.append(xI.α_all,
                                             xI.α+np.zeros(time_dead-1))
                        xI.β_all = np.append(xI.β_all,
                                             xI.β+np.arange(1,time_dead))
                        xI.n = np.append(xI.n,np.zeros(time_dead-1))
                        m_avgs,V_avgs = [],[]
                        for kk in range(1,time_dead):
                            mk = np.array([xI.up_m[c][kk,:] for c in range(Nc)])
                            Vk = np.array([xI.up_V[c][kk,kk]*np.eye(dims)
                                           for c in range(Nc)])
                            mk_avg,Vk_avg = collapse(mk,Vk,xI.up_π)
                            m_avgs.append(mk_avg)
                            V_avgs.append(Vk_avg)
                        m_avgs,V_avgs = np.array(m_avgs),np.array(V_avgs)
                        xI.X = np.vstack([xI.X,m_avgs[::-1,:]])
                        xI.Var = np.vstack([xI.Var,V_avgs[::-1,:,:]])
                        π_stack = (xI.π[-1:,:].T*np.ones(time_dead-1)).T
                        xI.π = np.vstack([xI.π,π_stack])
                    xI.ζ = [True] * len(xI.ζ)
                    xI.Ω = [False] * len(xI.ζ)
                    xI.κ_death = None
                    xI.X_best = xI.X_best[:max(len(xI.ζ)-d,0),:]
                    xI.Var_best = xI.Var_best[:max(len(xI.ζ)-d,0)]
                    # update track
                    xI.update(yι,s2)
                    # wipe track used for revival from records
                    self.tracks.pop(ι-R+1)
                    self.a[self.a==ι-R+2] = I
                    self.a[self.a>ι-R+2] -= 1
            
        # split 'counter'
        S = Nk - R
        
        # splits
        for i in range(Nk_1):
            xi = self.tracks[i]
            
            # skip if inactive / no assocs / was revived
            if not xi.ζ[-1] or xi.n[-1] == 0 or i in revived:
                continue
            
            # propose split
            k_κp = xi.get_k_minus_κ_prime(True)
            k_back = rng.choice(k_κp) + 1
            
            # i quantities
            yi = y[self.a==i+1,:]
            yibar = yi.mean(0)
            ni = yi.shape[0]
            Li = log_ρ + dims * np.log(2*np.pi*s2 / ni) / 2
            Li += norm.logpdf(yi,yibar,s2**0.5).sum()
            log_p_dif = np.zeros(Nk_1+1)
            
            # compute λκi
            for ι in range(Nk_1):
                xι = self.tracks[ι]
                if xι.Ω[-1] == 0 and ι != i:
                    # no contribution if ineligible for revival (and ≠ i)
                    log_p_dif[ι+1] = -np.inf
                    continue
                elif ι != i:
                    L_ιi = np.array([log_lik_offset_var(yi,xι.m_preds[c][0,:],
                                xι.V_preds[c][0,0],s2) for c in range(Nc)])
                    L_marg_ιi = np.log((np.exp(L_ιi)*xι.up_π).sum())
                    ψ_term = (k - xι.κ_death + 1) * log_ψ - log_1_ψ
                    µ_term = log_gamma_ratio(xι.α,ni) + xι.α * np.log(xι.β)
                    µ_term -= (xι.α + ni) * np.log(xι.β + k - xι.κ_death + 1)
                    µ_term -= log_gamma_ratio(αp,ni) + αp * np.log(βp)
                    µ_term += (αp + ni) * np.log(βp + 1)
                    γ_term = np.log(ηk+1-R) + np.log(ξk_1+1) - np.log(εk_1+ηk-R)
                    log_p_dif[ι+1] = L_marg_ιi - Li + ψ_term + µ_term + γ_term
                else:
                    L_ιi = np.array([log_lik_offset_var(yi,xι.m_preds[c][0,:],
                                xι.V_preds[c][0,0],s2) for c in range(Nc)])
                    L_marg_ιi = np.log((np.exp(L_ιi)*xι.π[-2,:]).sum())
                    ψ_term = k_back * log_ψ - log_1_ψ
                    µ_term = log_gamma_ratio(xι.α-ni,ni) - xι.α * np.log(xι.β)
                    µ_term += (xι.α - ni) * np.log(xι.β - k_back)
                    µ_term -= log_gamma_ratio(αp,ni) + αp * np.log(βp)
                    µ_term += (αp + ni) * np.log(βp + 1)
                    γ_term = np.log(ηk+1-R) + np.log(ξk_1+1) - np.log(εk_1+ηk-R)
                    log_p_dif[ι+1] = L_marg_ιi - Li + ψ_term + µ_term + γ_term
            λκi = k_κp / np.exp(log_p_dif).sum()
            
            # sample acceptance
            if rng.random() < λκi:
                # accept split
                S += 1
                # delete track i k_back steps ago
                xi.κ_death = k - k_back + 1
                xi.ζ[-k_back:] = [False] * k_back
                xi.Ω[-k_back:] = [True] * k_back
                xi.α -= ni
                xi.β -= k_back
                xi.α_all = xi.α_all[:-k_back]
                xi.β_all = xi.β_all[:-k_back]
                xi.n = xi.n[:-k_back]
                xi.up_π = xi.π[-2,:]
                xi.π = xi.π[:-k_back,:]
                xi.X = xi.X[:-k_back,:]
                xi.Var = xi.Var[:-k_back]
                m_avgs,V_avgs = [],[]
                for kk in range(k_back,xi.m_preds[0].shape[0]):
                    mk = [xi.m_preds[c][kk,:] for c in range(Nc)]
                    Vk = [xi.V_preds[c][kk,kk]*np.eye(dims) for c in range(Nc)]
                    πk = xi.up_π
                    mk_avg,Vk_avg = collapse(np.array(mk),np.array(Vk),πk)
                    m_avgs.append(mk_avg)
                    V_avgs.append(Vk_avg)
                m_avgs,V_avgs = np.array(m_avgs[::-1]),np.array(V_avgs[::-1])
                if xi.X_best.shape[0] > 0:
                    xi.X_best = np.vstack([xi.X_best[:-1,:],m_avgs])
                    xi.Var_best = np.vstack([xi.Var_best[:-1,:,:],V_avgs])
                else:
                    xi.X_best = m_avgs
                    xi.Var_best = V_avgs
                # initialise new track
                self.tracks.append(Track_iSE(yi,s2,k,self.track_props))
                self.a[self.a==i+1] = S
        warnings.filterwarnings("default", category=RuntimeWarning)       
        
        # finalise particle
        self.ε += S - Nk_1 - ηk
        self.η[-1] = S - Nk_1
        self.ε_all[-1] = self.ε
    
    
    # plot particle
    def plot(self,colours,alpha,line='-'):
        # add full tracks
        for i in range(len(self.tracks)):
            x = self.tracks[i]
            colour = colours[i%len(colours)]
            mkrs = ['x','.'] if x.ζ[-1] else ['x','+']
            x.plot(colour,line,mkrs,'',alpha)

##########################################



##### particle filter class #####


class ParticleFilter:
    def __init__(self,J,particle_props,track_props,scene_props):
        self.J = J # number of particles
        self.S = scene_props[1] # min / max values for x and y directions
        self.aspect = scene_props[2] # true aspect ratio, for plotting
        self.particles = [Particle(J,particle_props,track_props,scene_props,j)
                          for j in range(J)] # create J particles
        self.track_props = track_props # allow filter access to track properties
        
        # prepare quantities to compute real-time metrics, even after the fact
        self.filter_quantities = []
        self.full_quantities = []
        self.track_associations = []
        self.j_quantities = []
        self.weight_matrix = np.ones([0,J]) # save all weights
    
    
    # adaptive resampling ("bothered=False" allows resampling to be skipped)
    def resample(self,bothered=True):
        # weights
        weights = np.array([particle.w for particle in self.particles])        
        # check adaptive criterion (the effective number of particles)
        if not bothered or (weights**2).sum()**-1 > self.J / 10:
            # skip if no resampling wanted / needed
            return None
        new_particles = []
        # resample particles
        indices = rng.choice(self.J,self.J,True,weights)
        # populate 'new_particles'
        for i in indices:
            new_particles.append(dc(self.particles[i]))
        # reset weights and track IDs
        for j in range(len(new_particles)):
            new_particles[j].w = 1 / self.J
            new_particles[j].id = j
        # save new particles
        self.particles = new_particles
    
    
    # data kernel smoothing density (Gaussian pdfs at each yl with variance s2)
    def ksd(self,y,s2):
        n,_ = y.shape
        # negative ksd and its derivative
        def neg_pdf_and_dpdf(x):
            # make compatible for lists and 1d arrays as well as 2d arrays
            xr = x
            if type(xr) == list:
                xr = np.array(xr)
            if len(xr.shape) == 1:
                xr = xr.reshape([1,-1])
            # get densities for individual peaks
            densities = (norm.pdf(y,xr,s2**0.5).prod(1) / n).reshape([1,n])
            # get pdf values at all x
            pdf_val = densities.sum()
            # get derivative of pdf at all x
            dpdf_vec = (densities @ (y - xr)).sum(0) / s2
            return -pdf_val, -dpdf_vec.flatten()
        # return function
        return neg_pdf_and_dpdf
    
    
    # data clustering
    def clustering(self,y,num_sd=1):
        
        # useful quantities
        n,dim = y.shape
        
        # s2 estimate averaged over particles
        s2 = sum([p.w * p.b / (p.φ - 1) for p in self.particles])
        
        # make ksd
        neg_f_and_df = self.ksd(y,s2)
        
        # initialise results
        G = []
        peaks = np.zeros([0,dim])
        
        # cluster data
        for l in range(n):
            # optimise ksd at each datum yl
            res = minimize(neg_f_and_df,y[l,:],method='BFGS',jac=True,
                           options={'xrtol':s2**0.5/10})
            added_l = False # haven't added yl to any cluster yet
            for θ in range(peaks.shape[0]):
                if np.allclose(peaks[θ,:],res.x,0,num_sd*s2**0.5):
                    # add l to G_kθ if z_kl close to peak θ
                    G[θ].append(l)
                    # udpate mean of G_kθ
                    peaks[θ,:] = y[G[θ],:].mean(0)
                    added_l = True # added yl to G_kθ
                    break
            if added_l:
                continue
            # start new cluster if z_kl far away from all other peaks
            peaks = np.vstack([peaks,res.x.reshape([1,-1])])
            # add new peak to list of peaks
            G.append([l])
            
        # return c;usters
        return G
    
    
    # update weights
    def update_w(self,y):
        
        # useful quantities
        Nc = self.track_props[3]
        
        # compute terms for log(weight increment)
        old_log_w = np.array([np.log(p.w + 1e-100) for p in self.particles])
        log_Qch = np.array([p.log_Qch for p in self.particles])
        log_ψ_terms = np.array([sum([x.ζ_expo for x in p.tracks])*np.log(1-p.ψ)
                                for p in self.particles])
        log_P = np.zeros(self.J)
        log_Pch = np.zeros(self.J)
        for j in range(self.J):
            p = self.particles[j]
            s2 = p.b / (p.φ - 1)
            log_P[j] += (sum(p.a==0) + p.η[-1]) * np.log(p.ρ)
            for i in range(len(p.tracks)):
                x = p.tracks[i]
                if not x.ζ[-1]:
                    continue
                yi,ni = y[p.a==i+1,:],sum(p.a==i+1)
                if i < len(p.tracks) - p.η[-1]:
                    m1 = [x.m_preds[c][0,:] for c in range(Nc)]
                    V11 = [x.V_preds[c][0,0] for c in range(Nc)]
                    wN = [x.π[-2,c]*np.exp(log_lik_offset_var(yi,m1[c],V11[c],s2))
                          for c in range(Nc)]
                    log_P[j] += np.log(sum(wN)+1e-100)
                else:
                    log_P[j] += yi.shape[1] * np.log(2*np.pi*s2 / ni) / 2
                    log_P[j] += norm.logpdf(yi,yi.mean(0),s2**0.5).sum()
                log_Pch[j] += log_gamma_ratio(x.α_all[-2],ni)
                log_Pch[j] += x.α_all[-2]*np.log(x.β_all[-2]) - x.α*np.log(x.β)
            log_Pch[j] += log_gamma_ratio(p.α0_all[-2],sum(p.a==0))
            log_Pch[j] += p.α0_all[-2]*np.log(p.β0_all[-2]) - p.α0*np.log(p.β0)
            log_Pch[j] += log_gamma_ratio(p.ε_all[-2],p.η[-1])
            log_Pch[j] -= log_gamma_ratio(1,p.η[-1])
            log_Pch[j] += p.ε_all[-2]*np.log(p.ξ_all[-2]) - p.ε*np.log(p.ξ)
        
        # combine with old weights and normalise
        log_w = old_log_w + log_ψ_terms + log_P + log_Pch - log_Qch
        w = normalise(log_w)
        # save new weights to each particle
        for j in range(self.J):
            self.particles[j].w = w[j]
    
    
    # save filtering quantities for real-time metrics
    def save_filter(self):
        filter_quants = []
        for p in self.particles:
            target_quants = [[dc(x.up_m),dc(x.up_V),dc(x.κ_birth),dc(x.α),
                              dc(x.β),dc(x.up_π),dc(x.ζ)] for x in p.tracks]
            N_act = sum([int(x.ζ[-1]) for x in p.tracks])
            p_quants = [target_quants,dc(p.φ),dc(p.b),dc(p.ε),dc(p.ξ),dc(p.α0),
                        dc(p.β0),N_act,dc(p.w)]
            filter_quants.append(p_quants)
        self.filter_quantities.append(filter_quants)
    
    
    # save full trajectory quantities for real-time metrics
    def save_full(self):
        full_quants = []
        for p in self.particles:
            target_quants = [[dc(x.up_m),dc(x.up_V),dc(x.κ_birth),dc(x.ζ),
                              dc(x.α_all),(x.β_all),dc(x.π),dc(x.X),dc(x.Var),
                              dc(x.X_best),dc(x.Var_best)] for x in p.tracks]
            p_quants = [target_quants,dc(p.α0_all),dc(p.β0_all),dc(p.ε_all),
                        dc(p.ξ_all),dc(p.φ_all),dc(p.b_all),dc(p.w)]
            full_quants.append(p_quants)
        self.full_quantities.append(full_quants)
    
    
    # plot main scene
    def plot_scenario(self,truths,end_steps,y,track_colours,track_lines='--',
                      truth_colour='goldenrod',truth_line='-'):
        
        # useful quantities
        Tmax = len(y)
        
        # set aspect ratio
        plt.rcParams["figure.figsize"] = self.aspect
        
        # data
        for k in range(Tmax):
            plt.plot(y[k][:,0],y[k][:,1],'k2',alpha=0.01,label='')
        
        # ground truths
        for i in range(len(truths)):
            x,κ_end = truths[i],end_steps[i]
            plt.plot(x[:,0],x[:,1],c=truth_colour,ls=truth_line,label='')
            for end in [0,-1]:
                if end == 0:
                    mrk = 'x'
                elif κ_end < Tmax:
                    mrk = '+'
                else:
                    mrk = '.'
                plt.plot(x[end,0],x[end,1],c=truth_colour,ls='',marker=mrk,
                         label='')
        
        # opacities
        weights = np.array([p.w for p in self.particles])
        opacities = weights / weights.max()
        
        # tracks for all particles
        for j in range(self.J):
            p = self.particles[j]
            p.plot(track_colours,opacities[j],track_lines)
        
        # add lines for legend
        plt.plot(self.S[0,0]-10,0,'k2',alpha=0.1,label='Obs.')
        plt.plot(self.S[0,0]-10*np.ones(2),[0,1],c=truth_colour,ls=truth_line,
                 marker='',label='Truth')
        plt.plot(self.S[0,0]-10*np.ones(2),[0,1],c=track_colours[0],
                 ls=track_lines,marker='',label='Track')
        
        # tidy plot
        plt.xlim(self.S[0,:])
        plt.ylim(self.S[1,:])
        plt.legend(loc=0,framealpha=1)
        plt.show()
    
    
    # plot global hyperparameters
    def plot_global_hyperpars(self,true_s2=None,true_γ=None,true_µ0=None,
                              num_sds=3):
        
        # useful quantities
        steps = np.arange(len(self.filter_quantities))
        
        # set aspect ratio
        plt.rcParams["figure.figsize"] = [6,3]
        
        # prepare quantities
        s2_means = np.array([sum([P[-1]*P[2]/(P[1]-1) for P in FQ])
                             for FQ in self.filter_quantities])
        s2_sds = np.array([sum([P[-1]*P[2]**2/(P[1]-1)**2/(P[1]-2) 
                                for P in FQ])**0.5
                           for FQ in self.filter_quantities])
        γ_means = np.array([sum([P[-1]*P[3]/P[4] for P in FQ])
                            for FQ in self.filter_quantities])
        γ_sds = np.array([sum([P[-1]*P[3]/P[4]**2 for P in FQ])**0.5
                          for FQ in self.filter_quantities])
        µ0_means = np.array([sum([P[-1]*P[5]/P[6] for P in FQ])
                             for FQ in self.filter_quantities])
        µ0_sds = np.array([sum([P[-1]*P[5]/P[6]**2 for P in FQ])**0.5
                           for FQ in self.filter_quantities])
        
        # plot s2
        if true_s2 is not None:
            plt.plot([0,steps[-1]],[true_s2,true_s2],c='goldenrod',ls='--',
                     label=r'True $s^2$')
        plt.plot(steps,s2_means,c='teal',ls='-',label=r'$s^2$ Est.')
        plt.fill_between(steps,s2_means-num_sds*s2_sds,s2_means+num_sds*s2_sds,
                         color='teal',alpha=0.3,
                         label=f'$\pm{{{num_sds}}}$ s.d.s')
        plt.legend(loc=0,framealpha=1)
        plt.show()
        
        # plot γ
        if true_γ is not None:
            plt.plot([0,steps[-1]],[true_γ,true_γ],c='goldenrod',ls='--',
                     label=r'True $\gamma$')
        plt.plot(steps,γ_means,c='blueviolet',ls='-',label=r'$\gamma$ Est.')
        plt.fill_between(steps,γ_means-num_sds*γ_sds,γ_means+num_sds*γ_sds,
                         color='blueviolet',alpha=0.3,
                         label=f'$\pm{{{num_sds}}}$ s.d.s')
        plt.legend(loc=0,framealpha=1)
        plt.show()
        
        # plot µ0
        if true_µ0 is not None:
            plt.plot([0,steps[-1]],[true_µ0,true_µ0],c='goldenrod',ls='--',
                     label=r'True $\mu_0$')
        plt.plot(steps,µ0_means,c='pink',ls='-',label=r'$\mu_0$ Est.')
        plt.fill_between(steps,µ0_means-num_sds*µ0_sds,µ0_means+num_sds*µ0_sds,
                         color='pink',alpha=0.3,
                         label=f'$\pm{{{num_sds}}}$ s.d.s')
        plt.legend(loc=0,framealpha=1)
        plt.show()
    
    
    # track-to-truth association up to k
    def track_to_truth_k(self,k,truths,births):
        
        # useful quantities
        dt = self.track_props[2]
        particle_quants = self.full_quantities[k]
        weights = np.array([[p[-1] for p in particle_quants]])
        G_start = datetime(2015,8,6,8,15,0) # arbitrary time for step k=0 (a good day at the cricket!)
        max_dist = 10
        self.track_associations.append([])
        self.j_quantities.append([])
        
        # truths up to k
        truths_k,births_k = [],[]
        for i in range(len(truths)):
            if births[i] > k:
                continue
            end_i = min(truths[i].shape[0],k+1-births[i])
            truths_k.append(truths[i][:end_i+1])
            births_k.append(births[i])
        births_k = np.array(births_k,int)
        
        # truths ––> Stone Soup
        truthsSS = truth_to_SS(truths_k,births_k,dt=dt)
        
        for j in range(self.J):
            targets = particle_quants[j][0]
        
            # tracks per particle ––> Stone Soup
            xis = []
            for x in targets:
                if x[3][-1]: # i.e., "if x is active at step k:"
                    # construct best (i.e., smoothed) trajectory
                    xis.append(np.zeros(x[-4].shape))
                    mx,πx = np.array(x[0]),x[6][-1]
                    best_mxs = np.einsum('ijk,i->jk',mx,πx)[::-1,:]
                    if x[-2].shape[0] > 0:
                        xis[-1][:x[-2].shape[0]-1,:] = x[-2][:-1,:]
                        xis[-1][x[-2].shape[0]-1:,:] = best_mxs
                    else:
                        xis[-1] = best_mxs
                else:                    
                    # trajectory for inactive track (no more smoothing needed)
                    xis.append(dc(x[-2]))
            # birth times
            κis = np.array([x[2] for x in targets])
            # detection rate hyperparameters
            αis = np.array([x[4][-1] for x in targets])
            βis = np.array([x[5][-1] for x in targets])
            # class probabilities
            πis = np.array([x[6][-1] for x in targets])
            # stone soup tracks
            tracksSS = tracks_to_SS(xis,κis,dt=dt)
            
            # associate tracks
            T2T = TrackToTruth(max_dist,1,1)
            associations = T2T.associate_tracks(tracksSS,truthsSS)
            
            # extract associations
            track_assocs = -np.ones([k+1,len(tracksSS)],int) # use a_ki = -1 to mean 'no association for track i at step k'
            for i,track in zip(range(len(tracksSS)),tracksSS):
                for assoc in associations:
                    if track not in assoc.objects:
                        continue
                    for ii,truth in zip(range(len(truthsSS)),truthsSS):
                        if truth not in assoc.objects:
                            continue
                        # find start / end steps of association
                        start_stamp = assoc.time_range.start_timestamp
                        end_stamp = assoc.time_range.end_timestamp
                        start_step = int((start_stamp - G_start).seconds // dt)
                        end_step = int((end_stamp - G_start).seconds // dt)
                        # record the association
                        track_assocs[start_step:end_step+1,i] = ii
            
            # save track-to-truth associations for particle j
            self.track_associations[-1].append(track_assocs)
            # save most important quantities from particle j
            self.j_quantities[-1].append([xis,κis,αis,βis,πis])
        # save weights 
        self.weight_matrix = np.vstack([self.weight_matrix,weights])
    
    
    # plot local hyperparameters
    def plot_local_hyperpars(self,truths,true_µis,true_Cis,true_κis,num_sds=3,
                             truth_indices=None):
        
        # useful quantities
        Tmax = len(self.track_associations)
        w_mat = self.weight_matrix
        
        # set aspect ratio
        plt.rcParams["figure.figsize"] = [6,3]
        
        # make truth indices if not provided
        if truth_indices is None:
            truth_indices = np.arange(len(true_µis))
        true_N = len(truth_indices)
        
        # prepare estimate / error objects
        µi_mean = np.full([Tmax,self.J,true_N],np.nan)
        µi_var = np.full([Tmax,self.J,true_N],np.nan)
        Ci_est = np.full([Tmax,self.J,true_N],np.nan) # P(Ci = 1)
        Ci_var = np.full([Tmax,self.J,true_N],np.nan)
        
        # compute estimates / errors
        for k in range(Tmax):
            for j in range(self.J):
                track_assocs = self.track_associations[k][j]
                if track_assocs.shape[1] == 0:
                    continue
                [xis,κis,αis,βis,πis] = self.j_quantities[k][j]
                µ_means = αis / βis
                µ_vars = αis / βis**2
                for ii in range(true_N):
                    if k < true_κis[truth_indices[ii]]:
                        continue
                    # get all tracks associated to truth ii
                    AT = np.where(track_assocs[k,:]==truth_indices[ii])[0]
                    if AT.shape[0]:
                        µi_mean[k,j,ii] = µ_means[AT].mean()
                        µi_var[k,j,ii] = µ_vars[AT].mean()
                        Ci_est[k,j,ii] = πis[AT,1].mean()
                        Ci_var[k,j,ii] = (πis[AT,0] * πis[AT,1]).mean()
        
        # plot estimates / errors
        for ii in range(true_N):
            
            # useful quantities
            age_i = truths[truth_indices[ii]].shape[0]
            birth_i = true_κis[truth_indices[ii]]
            steps = np.arange(age_i) + birth_i
            true_µ = true_µis[truth_indices[ii]]
            true_C = true_Cis[truth_indices[ii]]
            wii = w_mat[birth_i:birth_i+age_i,:]
            µii = nan_avg(µi_mean[birth_i:birth_i+age_i,:,ii],wii)
            µsdii = nan_avg(µi_var[birth_i:birth_i+age_i,:,ii],wii)**0.5
            Cii = nan_avg(Ci_est[birth_i:birth_i+age_i,:,ii],wii)
            Csdii = nan_avg(Ci_var[birth_i:birth_i+age_i,:,ii],wii)**0.5
            label_ind = str(truth_indices[ii]+1)
            
            # plot µi
            plt.plot([steps[0],steps[-1]],[true_µ,true_µ],c='goldenrod',ls='--',
                     label=f'True $\mu_{{{label_ind}}}$')
            plt.plot(steps,µii,c='pink',ls='-',label=f'$\mu_{{{label_ind}}}$ Est.')
            plt.fill_between(steps,µii-num_sds*µsdii,µii+num_sds*µsdii,
                             color='pink',alpha=0.3,
                             label=f'$\pm{{{num_sds}}}$ s.d.s')
            plt.legend(loc=0,framealpha=1)
            plt.show()
            
            # plot Ci
            plt.plot([steps[0],steps[-1]],[0.5,0.5],'k:',alpha=0.2,label='')
            plt.plot([steps[0],steps[-1]],[true_C,true_C],c='goldenrod',ls='--',
                     label=f'True $C_{{{label_ind}}}$')
            plt.plot(steps,Cii,c='deeppink',ls='-',
                     label=f'$P(C_{{{label_ind}}}=1)$')
            plt.fill_between(steps,Cii-Csdii,Cii+Csdii,color='deeppink',
                             alpha=0.3,label=r'$\pm1$ s.d.')
            plt.legend(loc=0,framealpha=1)
            plt.ylim([-0.02,1.02])
            plt.show()
    
    
    # compute weighted SIAP metrics (track breaks done at the end)
    def weighted_siap(self,truths,births):
        
        # useful quantities
        weights = self.weight_matrix
        Tmax = weights.shape[0]
        Nc = self.track_props[3]
        N_truth = len(truths)
        dims = truths[0].shape[1]
        gones = births + [truth.shape[0] for truth in truths]
        
        # Σ J(k)
        J_sum = sum([truth.shape[0] for truth in truths])
        
        # atomic results array (sufficient for computing SIAP metrics)
        k_atoms = np.zeros([4,Tmax,self.J]) # JT / NA / N / PA
        NU_1,TT = np.zeros([N_truth,self.J]),np.zeros([N_truth,self.J])
        
        for k in range(Tmax):
            for j in range(self.J):
                
                # track-to-truth associations at k
                track_ak = self.track_associations[k][j][-1,:]
                # number of tracks at k (incl. deleted)
                Nk = track_ak.shape[0]
                # target quantities saved at k
                TQ = self.filter_quantities[k][j][0]
                Nk_active = self.filter_quantities[k][j][-2]
                
                # active tracks and their estimates
                active_tracks,active_locs = [],[]
                for i in range(Nk):
                    if TQ[i][-1][-1]:
                        # add index if track active
                        active_tracks.append(i)
                        πx = TQ[i][-2]
                        m0 = sum([TQ[i][0][c][0,:]*πx[c] for c in range(Nc)])
                        active_locs.append(m0)
                    else:
                        # add dummy point estimate, even if track inactive
                        active_locs.append(np.zeros(dims))
                active_locs = np.array(active_locs)
                
                # JT(k), NA(k) & PA(k)
                for i in range(N_truth):
                    if births[i] <= k < gones[i] and i in track_ak:
                        # count associated truth: JT(k)
                        k_atoms[0,k,j] += 1
                        # tracks associated to truth i: NA(k)
                        associated_tracks = np.where(track_ak==i)[0]
                        k_atoms[1,k,j] += associated_tracks.shape[0]
                        # distances for truth i: PA(k)
                        dif_i = active_locs[associated_tracks,:]
                        dif_i -= truths[i][k-births[i],:]
                        k_atoms[3,k,j] += ((dif_i**2).sum(1)**0.5).sum()
                
                # N(k)
                k_atoms[2,k,j] = Nk_active
                
        for j in range(self.J):
            
            # final quantities
            track_a1k = self.track_associations[-1][j]
            
            # NU_{truth} & TT_{truth}
            for i in range(N_truth):
                # associated tracks to truth i at each time step
                tracks_by_time = [np.where(track_a1k[k,:]==i)[0]
                                  for k in range(births[i],gones[i])]
                min_tracks_needed = 0
                step = 0
                while step < gones[i] - births[i]:
                    if tracks_by_time[step].shape[0] == 0:
                        # skip if untracked at step
                        step += 1
                    else:
                        # use another track
                        min_tracks_needed += 1
                        # longest track (continuously) associated to truth i from step
                        t_lens = np.ones(tracks_by_time[step].shape[0],int)
                        for ι_ind in range(tracks_by_time[step].shape[0]):
                            ι = tracks_by_time[step][ι_ind]
                            for k in range(step+1,gones[i]-births[i]):
                                if ι in tracks_by_time[k]:
                                    t_lens[ι_ind] += 1
                                else:
                                    break
                        # jump forward to end of longest track
                        step += t_lens.max()
                        # add number of steps tracked
                        TT[i,j] += t_lens.max()
                NU_1[i,j] = max(min_tracks_needed-1,0)
                
        # weighted atomic quantities
        JT_sum = (k_atoms[0,:,:] * weights).sum()
        NA_sum = (k_atoms[1,:,:] * weights).sum()
        N_sum = (k_atoms[2,:,:] * weights).sum()
        PA_sum = (k_atoms[3,:,:] * weights).sum()
        NU_1_sum = (NU_1 @ weights[-1:,:].T).sum()
        TT_sum = (TT @ weights[-1:,:].T).sum()
        
        # weighted SIAP
        C = JT_sum / J_sum if J_sum > 0 else 0
        A = NA_sum / JT_sum if JT_sum > 0 else 0
        S = 1 - NA_sum / N_sum if N_sum > 0 else 0 # valid by bracket expansion
        PA = PA_sum / NA_sum if NA_sum > 0 else 0
        R = NU_1_sum / TT_sum * 1000 if TT_sum > 0 else 0
        
        return C,A,S,PA,R
    
    
    # GOSPA on point estimates
    def GOSPA_pt_est(self,truths,births,c=10,p=2):
        
        if len(truths) == 0:
            print("haven't thought of this...probs easy but I don't need it")
            return None
        
        # useful quantities
        weights = self.weight_matrix
        Tmax = weights.shape[0]
        N_truth = len(truths)
        gones = births + [truth.shape[0] for truth in truths]
        dims = truths[0].shape[1]
        
        # compute GOSPA(k)^p for all k
        gospa_p = np.zeros(Tmax)
        for k in range(Tmax):
            # active truth objects at k
            X_true = []
            for i in range(N_truth):
                if births[i] <= k < gones[i]:
                    X_true.append(truths[i][k-births[i],:])
            # point estimate
            Xi = self.point_est(k,dims,c)
            Nk_active = Xi.shape[0]
            if len(X_true) > 0 and Nk_active > 0:
                # linear assignment cost matrix
                cost_M = np.array([np.min([((Xi-y)**2).sum(1)**0.5,
                                            c*np.ones(Nk_active)],0)**p
                                    for y in X_true]).T
                cost_M = cost_M.astype(float)
                # cost of solution
                min_rows,min_cols = linear_sum_assignment(cost_M)
                gospa_p[k] = cost_M[min_rows,min_cols].sum()
            gospa_p[k] += c**p * np.abs(Nk_active - len(X_true)) / 2
        
        # average GOSPA over time steps
        return (gospa_p ** (1/p)).mean()
    
    
    # obtain point estimate locations for all tracks and particles
    def point_est(self,k,dims,max_dist,min_w=1e-5,w_lim=0.5):
        
        # weights at k
        w = self.weight_matrix[k,:]
        
        # number of classes
        Nc = self.track_props[3]
        
        # target locations
        locs = []
        for j in range(self.J):
            TQ = self.filter_quantities[k][j][0]
            Nk = len(TQ)
            # active track locations at k for j
            locs_j = []
            for i in range(Nk):
                if TQ[i][-1][-1]:
                    # add track location, as i is active
                    πx = TQ[i][-2]
                    m0 = sum([TQ[i][0][c][0,:]*πx[c] for c in range(Nc)])
                    locs_j.append(m0)
            locs.append(np.array(locs_j))
        
        Nj = np.array([locs[j].shape[0] for j in range(self.J)],int)
        
        # prepare objects
        pot_targs,targ_means,targ_w = [],np.zeros([0,dims]),[]
        
        # find potential object clusters and their weights
        for j in range(self.J):
            if Nj.sum() == 0 or w[j] < min_w:
                continue
            for i in range(Nj[j]):
                if targ_means.shape[0] == 0:
                    pot_targs.append({(j,i)})
                    targ_means = np.vstack([targ_means,locs[j][i,:]])
                    targ_w.append(dc(w[j]))
                    continue
                d2 = ((targ_means - locs[j][i,:])**2).sum(1)
                ι = np.argmin(d2)
                d2_min = d2[ι]
                if d2_min > max_dist**2:
                    pot_targs.append({(j,i)})
                    targ_means = np.vstack([targ_means,locs[j][i,:]])
                    targ_w.append(dc(w[j]))
                    continue
                pot_targs[ι].add((j,i))
                targ_w[ι] += w[j]
                targ_means[ι] = np.array(sum([w[ji[0]]*locs[ji[0]][ji[1],:]
                                              for ji in pot_targs[ι]]))
                targ_means[ι] /= sum([w[ji[0]] for ji in pot_targs[ι]])
        
        targ_ests = []
        for ι in range(len(targ_w)):
            w_left = targ_w[ι]
            while w_left > w_lim:
                # repeatedly add target while weight is above threshold
                targ_ests.append(targ_means[ι,:])
                w_left -= 1
        
        return np.array(targ_ests)
    
    
    # hyperparameter metrics
    def hyper_metrics(self,s2,γ,µ0,all_µi,all_Ci,births,deaths):
        
        # useful quantities
        Tmax = len(self.track_associations)
        w_mat = self.weight_matrix
        true_N = all_µi.shape[0]
        
        # global quantities
        s2_means = np.array([sum([P[-1]*P[2]/(P[1]-1) for P in FQ])
                             for FQ in self.filter_quantities])
        s4s = np.array([sum([P[-1]*P[2]**2/(P[1]-1)/(P[1]-2) for P in FQ])
                        for FQ in self.filter_quantities])
        γ_means = np.array([sum([P[-1]*P[3]/P[4] for P in FQ])
                            for FQ in self.filter_quantities])
        γ2s = np.array([sum([P[-1]*P[3]*(P[3]+1)/P[4]**2 for P in FQ])
                        for FQ in self.filter_quantities])
        µ0_means = np.array([sum([P[-1]*P[5]/P[6] for P in FQ])
                             for FQ in self.filter_quantities])
        µ02s = np.array([sum([P[-1]*P[5]*(P[5]+1)/P[6]**2 for P in FQ])
                         for FQ in self.filter_quantities])
        
        # compute average RMSEs for global quantities
        avg_s2_rmse = ((s4s - 2*s2*s2_means + s2**2)**0.5).mean()
        avg_γ_rmse = ((γ2s - 2*γ*γ_means + γ**2)**0.5).mean()
        avg_µ0_rmse = ((µ02s - 2*µ0*µ0_means + µ0**2)**0.5).mean()
        
        # prepare estimate for objects
        µi_mean = np.full([Tmax,self.J,true_N],np.nan)
        µi2 = np.full([Tmax,self.J,true_N],np.nan)
        Ci_est = np.full([Tmax,self.J,true_N],np.nan)
        
        # compute estimates / errors
        for k in range(Tmax):
            for j in range(self.J):
                track_assocs = self.track_associations[k][j]
                if track_assocs.shape[1] == 0:
                    continue
                [xis,κis,αis,βis,πis] = self.j_quantities[k][j]
                µ_means = αis / βis
                µ2s = αis * (αis + 1) / βis**2
                for ii in range(true_N):
                    if k < births[ii]:
                        continue
                    # get all tracks associated to truth ii
                    AT = np.where(track_assocs[k,:]==ii)[0]
                    if AT.shape[0]:
                        µi_mean[k,j,ii] = µ_means[AT].mean()
                        µi2[k,j,ii] = µ2s[AT].mean()
                        Ci_est[k,j,ii] = πis[AT,1].mean()
        
        # prepare local RMSEs
        avg_µi_rmse = 0
        avg_Ci_rmse = 0
        track_time = 0
        
        # compute average RMSEs for local quantities
        for ii in range(true_N):
            
            # useful quantities
            true_µ = all_µi[ii]
            true_C = all_Ci[ii]
            wii = w_mat[births[ii]:deaths[ii],:]
            µii = nan_avg(µi_mean[births[ii]:deaths[ii],:,ii],wii)
            µ2ii = nan_avg(µi2[births[ii]:deaths[ii],:,ii],wii)
            Cii = nan_avg(Ci_est[births[ii]:deaths[ii],:,ii],wii)
            k_ii = µii.shape[0] - np.isnan(µii).sum()
            
            # compute RMSE for truth ii
            track_time += k_ii
            avg_µi_rmse += np.nansum((µ2ii - 2*true_µ*µii + true_µ**2)**0.5)
            Ci_rmses = np.abs((true_C * (1-Cii) + (1-true_C) * Cii))**0.5
            avg_Ci_rmse += np.nansum(Ci_rmses)
        
        # average over total tracking time
        if track_time != 0:
            avg_µi_rmse /= track_time
            avg_Ci_rmse /= track_time
        else:
            avg_µi_rmse,avg_Ci_rmse = np.nan,np.nan
        
        return avg_s2_rmse,avg_γ_rmse,avg_µ0_rmse,avg_µi_rmse,avg_Ci_rmse
    
    
    # save results (final tracks and metrics)
    def save(self,name,ground_truths):
        # NOTE: fixed-lag smoothed estimates saved for tracks, but not metrics!
        
        # most likely particle
        weights = np.array([p.w for p in self.particles])
        best_j = np.where(weights==max(weights))[0][0]
        best_particle = self.particles[best_j]
        
        # construct and record tracks from best particle (highest weight)
        tracks,births = [],[]
        for track in best_particle.tracks:
            births.append(track.κ_birth)
            if track.ζ[-1]:
                mx,πx = np.array(track.up_m),track.up_π
                best_mxs = np.einsum('ijk,i->jk',mx,πx)[::-1,:]
                if track.X_best.shape[0] > 0:
                    x_best = np.vstack([track.X_best[:-1,:],best_mxs])
                else:
                    x_best = best_mxs
            else:
                x_best = track.X_best
            tracks.append(x_best)
        tracks,births = np.array(tracks,object),np.array(births,int)
        
        # metrics
        truths,true_births,true_deaths,s2,γ,µ0,µis,classes = ground_truths
        metrics = np.zeros(11)
        metrics[:5] = self.weighted_siap(truths,true_births)
        metrics[5] = self.GOSPA_pt_est(truths,true_births)
        metrics[6:] = self.hyper_metrics(s2,γ,µ0,µis,classes,true_births,
                                         true_deaths)
        
        np.savez(name,tracks=tracks,births=births,metrics=metrics)
        
        return metrics


#################################
















##### iSE functions #####

# integral of Gaussian cdf
def Xi(v,w,l):
    
    # make vector / number inputs all valid to use
    v = np.array([v]).reshape([np.max(np.array([v]).shape),1])
    w = np.array([w]).reshape([1,np.max(np.array([w]).shape)])
    
    # compute useful sizes
    n = v.shape[0]
    m = w.shape[1]
    
    if l>1e-100:
        
        # compute sufficient n x m matrices for vectorised computations
        v_sq = v * np.ones([1,m])
        w_sq = w * np.ones([n,1])
        dif_sq = v_sq - w_sq
        
        # combine to make kernel formula
        K = dif_sq*norm.cdf(dif_sq/l) + l**2 * norm.pdf(dif_sq,0,l)
        K *= (2*np.pi)**0.5 * l
        
        return K
    
    else:
        
        # reject if l too small
        print('\n l may be too small to compute this matrix \n')
        
        return np.zeros([n,m])



# iSE function
def iSE(v,w,s2,l):
    
    # make vector / number inputs all valid to use
    v = np.array([v]).reshape([np.max(np.array([v]).shape),1])
    w = np.array([w]).reshape([1,np.max(np.array([w]).shape)])
    
    if l>1e-100:
        
        # make 't0' vector for each of v and w
        v0,w0 = np.zeros(v.shape),np.zeros(w.shape)
        # compute K in terms of Xi –– √(2π)𝓁 factor included in Xi function
        K = Xi(v,w0,l) + Xi(v0,w,l) - Xi(v,w,l) - l**2
        
        return s2 * (K + K.T) / 2
    
    else:
        # reject if l too small
        return None

#########################













##### metric functions #####


# make ground truth tracks Stone Soup compatible
def truth_to_SS(L,st,G_start=datetime(2015,8,6,8,15,0),dt=1):
    start_times = []
    for i in range(st.shape[0]):
        start_times.append(G_start + timedelta(seconds=int(dt*st[i])))
    all_tracks = []
    for i in range(len(L)):
        track = []
        for t in range(L[i].shape[0]):
            ts = start_times[i] + timedelta(seconds = dt * t)
            track.append(GroundTruthState(L[i][t,:],timestamp = ts))
        track = GroundTruthPath(track,'truth' + str(i))
        all_tracks.append(track)
    return all_tracks


# make inferred tracks Stone Soup compatible
def tracks_to_SS(L,st,G_start=datetime(2015,8,6,8,15,0),dt=1):
    start_times = []
    for i in range(st.shape[0]):
        start_times.append(G_start + timedelta(seconds=int(dt*st[i])))
    all_tracks = []
    for i in range(len(L)):
        track = Track(None,'track' + str(i))
        for t in range(L[i].shape[0]):
            ts = start_times[i] + timedelta(seconds = dt * t)
            track.append(State(L[i][t,:],timestamp = ts))
        all_tracks.append(track)
    return all_tracks


# extract PHD(-CV) tracks from stone soup
def phd_to_tracks(tracksSS,G_start,dt=1):
    # trivial when no tracks
    if len(tracksSS) == 0:
        return [],np.zeros(0,int)
    # prepare objects
    for track in tracksSS:
        dims = track[0].state_vector.shape[0] // 2
        break
    G_start = datetime(2015,8,6,8,15,0)
    tracks = []
    births = []
    for i,track in enumerate(tracksSS):
        # get specified locations
        steps = [int((state.timestamp-G_start).seconds//dt)
                 for state in track.states]
        steps = np.array(steps,int)
        states = [state.state_vector[np.arange(0,2*dims,2),0]
                  for state in track.states]
        births.append(steps[0])
        Xi = np.zeros([steps[-1]-steps[0]+1,dims])
        Xi[steps-steps[0],:] = states
        missed_steps = list(set(range(steps[0],steps[-1]+1)).difference(steps))
        # interpolate gaps
        for k in missed_steps:
            k_down = steps[steps<k][-1]
            Xi_down = Xi[k_down-steps[0],:]
            k_up = steps[steps>k][0]
            Xi_up = Xi[k_up-steps[0],:]
            Xi[k-steps[0],:] = ((k-k_down)*Xi_up + (k_up-k)*Xi_down)
            Xi[k-steps[0],:] /= (k_up - k_down)
        tracks.append(Xi)
    births = np.array(births,int)
    return tracks,births


# tabulate metrics to 'rounder' decimal places
def value_table(methods,metric_names,metrics,rounder=3):
    avg_metrics = metrics.mean(0).round(rounder)
    num_methods = len(methods)
    table = pd.DataFrame({methods[m]:avg_metrics[m,:] 
                          for m in range(num_methods)},index=metric_names)
    print('\n')
    print('Metrics')
    print(table)


# metrics for MATLAB-based methods (i.e., the message passing ones)
def metrics_matlab(Tmax,tracks,births,metric_quantities,c=10,p=2):
    
    # obtain metrics
    N = len(tracks)
    max_dist = 10
    truths,true_births,dt = metric_quantities
    N_truth = len(truths)
    if N_truth == 0:
        print("i still haven't thought about this...")
        return None
    dims = truths[0].shape[1]
    truthsSS = tracks_to_SS(truths,true_births,dt=dt)
    tracksSS = tracks_to_SS(tracks,births,dt=dt)
    
    # associate tracks
    T2T = TrackToTruth(max_dist,1,1)
    associations = T2T.associate_tracks(tracksSS,truthsSS)
    
    # extract associations
    G_start = datetime(2015,8,6,8,15,0)
    track_assocs = -np.ones([Tmax,N],int)
    for i,track in zip(range(N),tracksSS):
        for assoc in associations:
            if track not in assoc.objects:
                continue
            for ii,truth in zip(range(N_truth),truthsSS):
                if truth not in assoc.objects:
                    continue
                start_stamp = assoc.time_range.start_timestamp
                end_stamp = assoc.time_range.end_timestamp
                start_step = (start_stamp - G_start).seconds // dt
                end_step = (end_stamp - G_start).seconds // dt
                track_assocs[start_step:end_step+1,i] = ii
    
    ### GOSPA & SIAP
    deaths = births + [track.shape[0] for track in tracks]
    true_deaths = true_births + [truth.shape[0] for truth in truths]
    
    # atomic results array
    k_atoms = np.zeros([4,Tmax]) # JT / NA / N / PA (= RSS)
    NU_1,TT = np.zeros(N_truth),np.zeros(N_truth)
    gospa = np.zeros(Tmax)
    
    for k in range(Tmax):
            
        # active truth objects at k
        X_true = []
        for i in range(N_truth):
            if true_births[i] <= k < true_deaths[i]:
                X_true.append(truths[i][k-true_births[i],:])
        
        # active tracks and their estimates
        active_tracks,X,X_active = [],np.zeros([N,dims]),[]
        for i in range(N):
            if births[i] <= k < deaths[i]:
                # add index if track active
                active_tracks.append(i)
                # add point estimate
                X[i,:] = tracks[i][k-births[i],:]
                X_active.append(X[i,:])
        X_active = np.array(X_active)
        Nk = len(active_tracks)
        
        # GOSPA at k
        if len(X_true) > 0 and Nk > 0:
            # linear assignment cost matrix
            cost_M = np.array([np.min([((X_active-y)**2).sum(1)**0.5,
                                       c*np.ones(Nk)],0)**p
                               for y in X_true]).T
            # cost of solution
            min_rows,min_cols = linear_sum_assignment(cost_M)
            gospa[k] = cost_M[min_rows,min_cols].sum()
        gospa[k] += c**p * np.abs(Nk - len(X_true)) / 2
        gospa[k] = gospa[k] ** (1/p)
        
        # JT(k), NA(k) & PA(k)
        for i in range(N_truth):
            if true_births[i] <= k < true_deaths[i] and i in track_assocs[k,:]:
                # count associated truth: JT(k)
                k_atoms[0,k] += 1
                # tracks associated to truth i: NA(k)
                associated_tracks = np.where(track_assocs[k,:]==i)[0]
                k_atoms[1,k] += associated_tracks.shape[0]
                # distance for truth i: PA(k)
                dif_i = X[associated_tracks,:]
                dif_i -= truths[i][k-true_births[i],:]
                k_atoms[3,k] += ((dif_i**2).sum(1)**0.5).sum()
        
        # N(k)
        k_atoms[2,k] = len(active_tracks)
    
    # NU_{truth} & TT_{truth}
    for i in range(N_truth):
        # associated tracks to truth i at each time step
        tracks_by_time = [np.where(track_assocs[k,:]==i)[0]
                          for k in range(true_births[i],true_deaths[i])]
        min_tracks_needed = 0
        step = 0
        while step < true_deaths[i] - true_births[i]:
            if tracks_by_time[step].shape[0] == 0:
                # skip if untracked at step
                step += 1
            else:
                # use another track
                min_tracks_needed += 1
                # longest track (continuously) associated to truth i from step
                t_lens = np.ones(tracks_by_time[step].shape[0],int)
                for ι_ind in range(tracks_by_time[step].shape[0]):
                    ι = tracks_by_time[step][ι_ind]
                    for k in range(step+1,true_deaths[i]-true_births[i]):
                        if ι in tracks_by_time[k]:
                            t_lens[ι_ind] += 1
                        else:
                            break
                # jump forward to end of longest track
                step += t_lens.max()
                # add number of steps tracked
                TT[i] += t_lens.max()
        NU_1[i] = max(min_tracks_needed-1,0)
            
    # atomic quantities summed over time steps
    J_sum = sum([truth.shape[0] for truth in truths])
    JT_sum = k_atoms[0,:].sum()
    NA_sum = k_atoms[1,:].sum()
    N_sum = k_atoms[2,:].sum()
    PA_sum = k_atoms[3,:].sum()
    NU_1_sum = NU_1.sum()
    TT_sum = TT.sum()
    
    # final SIAP metrics
    C = JT_sum / J_sum
    A = NA_sum / JT_sum
    S = 1 - NA_sum / N_sum # valid by bracket expansion
    PA = PA_sum / NA_sum
    R = NU_1_sum / TT_sum * 1000 # to see values better
    
    # average GOSPA
    avg_gospa = gospa.mean()
    
    return np.array([C,A,S,PA,R,avg_gospa])


# compute stone soup metrics and save results
def saveSS(name,tracksSS,metric_quantities,c=10,p=2):
    
    # obtain metrics
    N = len(tracksSS)
    truths,true_births,Tmax,dt = metric_quantities
    N_truth = len(truths)
    if N_truth == 0:
        print("i still haven't thought about this...")
        return None
    G_start = datetime(2015,8,6,8,15,0)
    dims = truths[0].shape[1]
    truthsSS = tracks_to_SS(truths,true_births,dt=dt)
    
    # extract tracks from stone soup
    tracks,births = phd_to_tracks(tracksSS,G_start,dt)
    
    # make consistent with other stone soup tracks
    tracksSS_new = tracks_to_SS(tracks,births,dt=dt)
    
    # associate tracks
    max_dist = 10
    T2T = TrackToTruth(max_dist,1)
    associations = T2T.associate_tracks(tracksSS_new,truthsSS)
    
    # extract associations
    track_assocs = -np.ones([Tmax,N],int)
    for i,track in zip(range(N),tracksSS_new):
        for assoc in associations:
            if track not in assoc.objects:
                continue
            for ii,truth in zip(range(N_truth),truthsSS):
                if truth not in assoc.objects:
                    continue
                start_stamp = assoc.time_range.start_timestamp
                end_stamp = assoc.time_range.end_timestamp
                start_step = (start_stamp - G_start).seconds // dt
                end_step = (end_stamp - G_start).seconds // dt
                track_assocs[start_step:end_step+1,i] = ii
    
    ### GOSPA & SIAP
    deaths = births + [track.shape[0] for track in tracks]
    true_deaths = true_births + [truth.shape[0] for truth in truths]
    
    # atomic results array
    k_atoms = np.zeros([4,Tmax]) # JT / NA / N / PA
    NU_1,TT = np.zeros(N_truth),np.zeros(N_truth)
    gospa = np.zeros(Tmax)
    
    for k in range(Tmax):
            
        # active truth objects at k
        X_true = []
        for i in range(N_truth):
            if true_births[i] <= k < true_deaths[i]:
                X_true.append(truths[i][k-true_births[i],:])
        
        # active tracks and their estimates
        active_tracks,X,X_active = [],np.zeros([N,dims]),[]
        for i in range(N):
            if births[i] <= k < deaths[i]:
                # add index if track active
                active_tracks.append(i)
                # add point estimate
                X[i,:] = tracks[i][k-births[i],:]
                X_active.append(X[i,:])
        X_active = np.array(X_active)
        Nk = len(active_tracks)
        
        # GOSPA at k
        if len(X_true) > 0 and Nk > 0:
            # linear assignment cost matrix
            cost_M = np.array([np.min([((X_active-y)**2).sum(1)**0.5,
                                       c*np.ones(Nk)],0)**p
                               for y in X_true]).T
            # cost of solution
            min_rows,min_cols = linear_sum_assignment(cost_M)
            gospa[k] = cost_M[min_rows,min_cols].sum()
        gospa[k] += c**p * np.abs(Nk - len(X_true)) / 2
        gospa[k] = gospa[k] ** (1/p)
        
        # JT(k), NA(k) & PA(k)
        for i in range(N_truth):
            if true_births[i] <= k < true_deaths[i] and i in track_assocs[k,:]:
                # count associated truth: JT(k)
                k_atoms[0,k] += 1
                # tracks associated to truth i: NA(k)
                associated_tracks = np.where(track_assocs[k,:]==i)[0]
                k_atoms[1,k] += associated_tracks.shape[0]
                # RMSE for truth i: PA(k)
                dif_i = X[associated_tracks,:]
                dif_i -= truths[i][k-true_births[i],:]
                k_atoms[3,k] += ((dif_i**2).sum(1)**0.5).sum()
        
        # N(k)
        k_atoms[2,k] = len(active_tracks)
    
    # NU_{truth} & TT_{truth}
    for i in range(N_truth):
        # associated tracks to truth i at each time step
        tracks_by_time = [np.where(track_assocs[k,:]==i)[0]
                          for k in range(true_births[i],true_deaths[i])]
        min_tracks_needed = 0
        step = 0
        while step < true_deaths[i] - true_births[i]:
            if tracks_by_time[step].shape[0] == 0:
                # skip if untracked at step
                step += 1
            else:
                # use another track
                min_tracks_needed += 1
                # longest track (continuously) associated to truth i from step
                t_lens = np.ones(tracks_by_time[step].shape[0],int)
                for ι_ind in range(tracks_by_time[step].shape[0]):
                    ι = tracks_by_time[step][ι_ind]
                    for k in range(step+1,true_deaths[i]-true_births[i]):
                        if ι in tracks_by_time[k]:
                            t_lens[ι_ind] += 1
                        else:
                            break
                # jump forward to end of longest track
                step += t_lens.max()
                # add number of steps tracked
                TT[i] += t_lens.max()
        NU_1[i] = max(min_tracks_needed-1,0)
            
    # atomic quantities summed over time steps
    J_sum = sum([truth.shape[0] for truth in truths])
    JT_sum = k_atoms[0,:].sum()
    NA_sum = k_atoms[1,:].sum()
    N_sum = k_atoms[2,:].sum()
    PA_sum = k_atoms[3,:].sum()
    NU_1_sum = NU_1.sum()
    TT_sum = TT.sum()
    
    # final SIAP metrics
    C = JT_sum / J_sum if J_sum > 0 else 0
    A = NA_sum / JT_sum if JT_sum > 0 else 0
    S = 1 - NA_sum / N_sum if N_sum > 0 else 0
    PA = PA_sum / NA_sum if NA_sum > 0 else 0
    R = NU_1_sum / TT_sum * 1000  if TT_sum > 0 else 0
    
    # average GOSPA
    avg_gospa = gospa.mean()
    
    # synthesise metrics
    metrics = np.array([C,A,S,PA,R,avg_gospa])
    
    # save results
    np.savez(name,tracks=np.array(tracks,object),births=births,metrics=metrics)
    
    return metrics
    

# open results of any format
def open_results(method,set_num,metric_quantities):
    
    # GM-PHD
    if method == 'GM-PHD':
        file = np.load(f'Results/phd{set_num}.npz',allow_pickle=True)
        tracks,births = list(file['tracks']),file['births']
        metrics = file['metrics']
        
    # GNN-CV
    elif method == 'GNN-CV':
        file = np.load(f'Results/gnn{set_num}.npz',allow_pickle=True)
        tracks,births = list(file['tracks']),file['births']
        metrics = file['metrics']
    
    # DiGiT
    elif method == 'DiGiT':
        file = np.load(f'Results/digit{set_num}.npz',allow_pickle=True)
        tracks,births = list(file['tracks']),file['births']
        metrics = np.delete(file['metrics'],5)
    
    # MP with IMM model
    elif method == 'MP-IMM':
        
        # format results
        mp_imm = loadmat(f'Results/MP_IMM{set_num}.mat')
        MPtracks = mp_imm['estimatedTracks']
        changes = np.vstack([1-np.isnan(MPtracks[0,:,:]),
                             np.zeros([1,MPtracks.shape[2]])])
        changes -= np.vstack([np.zeros([1,MPtracks.shape[2]]),
                              1-np.isnan(MPtracks[0,:,:])])
        tracks,times = [],np.array([],dtype=int)
        Tmax = MPtracks.shape[1]
        for i in range(MPtracks.shape[2]):
            starts = np.where(changes[:,i]==1)[0]
            stops = np.where(changes[:,i]==-1)[0]
            times = np.append(times,starts)
            for t in range(starts.shape[0]):
                tracks.append(MPtracks[:2,starts[t]:stops[t],i].T)
        tracks = [tracks[i] for i in np.argsort(times)]
        births = times[np.argsort(times)]
        metrics = metrics_matlab(Tmax,tracks,births,metric_quantities)
    
    # standard MP
    elif method == 'MP-CV':
        
        # format results
        mp_imm = loadmat(f'Results/MP_CV{set_num}.mat')
        MPtracks = mp_imm['estimatedTracks']
        changes = np.vstack([1-np.isnan(MPtracks[0,:,:]),
                             np.zeros([1,MPtracks.shape[2]])])
        changes -= np.vstack([np.zeros([1,MPtracks.shape[2]]),
                              1-np.isnan(MPtracks[0,:,:])])
        tracks,times = [],np.array([],dtype=int)
        Tmax = MPtracks.shape[1]
        for i in range(MPtracks.shape[2]):
            starts = np.where(changes[:,i]==1)[0]
            stops = np.where(changes[:,i]==-1)[0]
            times = np.append(times,starts)
            for t in range(starts.shape[0]):
                tracks.append(MPtracks[:2,starts[t]:stops[t],i].T)
        tracks = [tracks[i] for i in np.argsort(times)]
        births = times[np.argsort(times)]
        metrics = metrics_matlab(Tmax,tracks,births,metric_quantities)
    
    # GaPP-ReaCtion
    elif method == 'GaPP-ReaCtion':
        file = np.load(f'Results/GaPP_ReaCtion{set_num}.npz',allow_pickle=True)
        tracks,births = list(file['tracks']),file['births']
        metrics = file['metrics']
    
    # GaPP-Class
    elif method == 'GaPP-Class':
        file = np.load(f'Results/GaPP_Class{set_num}.npz',allow_pickle=True)
        tracks,births = list(file['tracks']),file['births']
        metrics = file['metrics']
    
    # unknown method
    else:
        raise NotImplementedError(f'unknown method: {method}')        
    
    return tracks,births,metrics
    
    
# plot results for method and data set
def plot_results(data,truths,true_deaths,tracks,deaths,scene,colours,method,
                 aspect=[6,3],wanting_legend=True):
    
    # set aspect ratio
    plt.rcParams["figure.figsize"] = aspect
    
    # prep
    cols = [colours] if type(colours) == str else colours
    num_cols = len(cols)
    
    # data
    for k in range(len(data)):
        plt.plot(data[k][:,0],data[k][:,1],'k2',alpha=0.05,label='')
        
    # truths
    for ι in range(len(truths)):
        m = '.' if true_deaths[ι] == len(data) else '+'
        plt.plot(truths[ι][:,0],truths[ι][:,1],c='goldenrod',ls='-',label='')
        plt.plot(truths[ι][0,0],truths[ι][0,1],c='goldenrod',ls='',marker='x',
                 label='')
        plt.plot(truths[ι][-1,0],truths[ι][-1,1],c='goldenrod',ls='',marker=m,
                 label='')
        
    # tracks
    for i in range(len(tracks)):
        m = '.' if deaths[i] == len(data) else '+'
        c = cols[i%num_cols]
        plt.plot(tracks[i][:,0],tracks[i][:,1],c=c,ls='--',label='')
        plt.plot(tracks[i][0,0],tracks[i][0,1],c=c,ls='',marker='x',label='')
        plt.plot(tracks[i][-1,0],tracks[i][-1,1],c=c,ls='',marker=m,label='')
        
    # legend
    if wanting_legend:
        plt.plot(scene[0,0]-5,scene[1,0]-5,'k2',alpha=0.2,label='Obs.')
        plt.plot(scene[0,0]-5,scene[1,0]-5,c='goldenrod',ls='-',label='Truth')
        plt.plot(scene[0,0]-5,scene[1,0]-5,c=cols[0],ls='--',label=method)
        plt.legend(loc=0,framealpha=1)
    
    # tidy plot
    plt.xlim(scene[0,:])
    plt.ylim(scene[1,:])
    plt.show()

############################















##### other functions #####

# iterate particle of GaPP-Class
def iterate(particle,y,G,k):
    particle.survival()
    particle.predict()
    particle.associate(y,G)
    particle.update(y,k)
    return particle


# perform revival on particle (makes GaPP-Class into GaPP-ReaCtion)
def revival(particle,y,k):
    particle.revival(y,k)
    return particle


# normalise weights
def normalise(w,input_log=True,output_log=False):
    if not input_log:
        log_w = np.log(w + 1e-100)
    else:
        log_w = dc(w)
    log_w -= log_w.max()
    weights = np.exp(log_w) / np.exp(log_w).sum()
    if output_log:
        return np.log(weights)
    else:
        return weights


# efficient log likelihood computation
def log_lik_offset_var(y,μ,V,Σ):
    
    # do nothing if no data
    if y.shape[0] == 0:
        return 0
    
    # useful quantities
    n,dims = y.shape
    ybar = y.mean(0)
    log_det = (n-1)*np.log(Σ) + np.log(Σ + n*V)
    std_err2 = ((y - ybar)**2).sum() / n
    mean_err2 = ((ybar - µ)**2).sum()
    
    # constant
    const = -dims * (n*np.log(2*np.pi) + log_det) / 2
    # exponent
    log_exp = -n * (std_err2/Σ + mean_err2/(Σ + n*V)) /  2
        
    return const + log_exp


# compute log(Γ(r+n)/Γ(r))
def log_gamma_ratio(r,n):
    
    # log(Γ(r+n)/Γ(r))
    LGR = np.log(np.arange(n) + r).sum()
    
    # note: to get log(Γ(m)), use r,n = 1,m-1
    return LGR


# collapse Gaussian mixture (Σ must be full matrix!)
def collapse(µ,Σ,π):
    N = π.shape[0]
    m = sum([µ[c] * π[c] for c in range(N)])
    µ_m = [(µ[c] - m).reshape([-1,1]) for c in range(N)]
    V = sum([π[c] * (Σ[c] + µ_m[c] @ µ_m[c].T) for c in range(N)])
    return m,V


# weighted average excluding nan (require dim(a) ∈ {1,2})
def nan_avg(a,weights=None):
    
    # format inputs
    if len(a.shape) == 1:
        A = a.reshape([1,-1])
    elif len(a.shape) == 2:
        A = a
    else:
        raise TypeError("dimension of 'a' should be 1 or 2")
    if weights is None:
        weights = np.ones(A.shape) / A.shape[1]
    
    # renormalise w based on nan values
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    w = ((A / A) * weights) / np.nansum((A / A) * weights,1,keepdims=True)
    
    # nan average
    nan_average = np.nansum(A * w,1)
    nan_average[np.isnan(A).all(1)] = np.nan
    warnings.filterwarnings("default", category=RuntimeWarning)

    return nan_average


# time into minutes (cannot go for >1 day)
def time_taken(start_time,end_time):
    if end_time < start_time:
        end_time += 24*60*60
    time = end_time - start_time
    mins = int(time // 60)
    secs = round(time - 60*mins,3)
    if mins == 0:
        return f'{secs} seconds'
    elif mins == 1:
        return f'1 minute and {secs} seconds'
    return f'{mins} minutes and {secs} seconds'

###########################