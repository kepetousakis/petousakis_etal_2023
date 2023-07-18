#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 13:59:56 2017

@author: papoutsi
"""
import numpy as np
#from scipy.stats import norm
from scipy.optimize import curve_fit

def spike_count(data):
    if data[0]>0:
        data[0]=-1
        
    # Find values above zero
    id_zero=np.nonzero(data>0)[0]
    if np.size(id_zero)==0:
        number_of_spikes=0 
        spike_timing=np.nan
    else:
        init_spike=id_zero[data[id_zero-1]<=0]
        if id_zero[-1]==np.size(data):
            np.append(data,-1)
          
    term_spike=id_zero[data[id_zero+1]<=0]
    spike_timing=np.round(np.mean([term_spike, init_spike], axis=0))
    number_of_spikes=np.size(spike_timing)
    
    return (spike_timing, number_of_spikes)

def normalization(data, mode):
    #  Mode 1: Mean-Variance normalization
    #   Mode 2: Min-Max normilization
    if (mode==1):      
        a=np.mean(data);
        b=np.std(data);
        norm_data= (data-a)/b; 
        if (b==0):
            norm_data=data;

    if (mode==2):
        a=np.max(data)
        b=np.min(data)
        norm_data= (data-b)/(a-b) 
        if ((a-b)==0):
            norm_data=data;
    return (norm_data)

def gauss(x,*p):
    A,mu,sigma=p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    

def tuning_properties(data,X):
    #Function to identify preferred orientation and calculate OSI, tuning
    #width.
    #Find orientation with max firing frequency.
    id_pref=np.nonzero(data==np.max(data))
    # print(data,X, id_pref)
    #Handling of cases with multiple peaks.
    if (np.size(id_pref)>1):
        #If their difference is less than 90 deg.
        if (id_pref[0][-1]-id_pref[0][0]<9):
            id_pref=np.int(np.ceil(np.mean(id_pref)));
        else:
            id_pref=id_pref[0][0];
    else:
        id_pref=id_pref[0][0]

    # Re-arrange to have a centered gaussian for fitting.
    id_X=np.size(X)/2
    if (id_pref<id_X+1):
        #new_x=np.concatenate((np.arange(id_pref+1+id_X,np.size(X)), np.arange(0,id_pref+id_X+1)), axis=0)
        new_x=np.concatenate((np.arange(id_pref+id_X,np.size(X)), np.arange(0,id_pref+id_X)), axis=0)
        new_x = np.array([int(x) for x in new_x])
        new_r=np.array([data[int(x)] for x in new_x])
        X_temp=np.array([X[int(x)] for x in new_x])
    else:
        new_x=np.concatenate((np.arange(id_pref-id_X,np.size(X)), np.arange(0,id_pref-id_X)), axis=0)
        new_r=np.array([data[int(x)] for x in new_x])
        X_temp=np.array([X[int(x)] for x in new_x])
    
    # Min-max normilazation of curves
    temp=normalization(new_r,2)
    # Define inital value for width (fitting is sensitive to inital width condition)
    # Starting fro your max value, find crossings to 0.5 (normalized curve)
 
    init_pref=np.where(X_temp==X[id_pref])[0][0]
    if (np.where(temp[init_pref+1:]<=0.5)[0].size==0):
        # ii_a=X.size
        ii_a = len(X)
    else:
        ii_a=np.where(temp[init_pref+1:]<=0.5)[0][0]   
    if (np.where(temp[:init_pref]<=0.5)[0].size==0):
        ii_b=0
    else:
        ii_b=np.where(temp[:init_pref]<=0.5)[0][-1]
    init_width=ii_a+init_pref+1-ii_b
    np.where(temp[:9]<=0.5)[-1]
    np.where(temp[:9]<=0.5)[-1]
    # Fit tuning curve
    p0=[1, init_pref*10,init_width*10]
    #p0=[1, X[id_pref], 20]
    try:
        coeff,var_matrix=curve_fit(gauss,X,temp, p0=p0)
    except Exception:#except RuntimeError:
        coeff=[1,np.where(X_temp==X[id_pref])[0][0]*10,360];
    if sum(gauss(X,*coeff)) == 0:  # some coeffs (e.g. [ 7.41454052e-01  5.87248218e+01 -1.88779460e-02] cause the gauss() function to return zeroes only)
        coeff=[1,np.where(X_temp==X[id_pref])[0][0]*10,360];
    fit=gauss(X,*coeff)
    #print(coeff, X)
    
    # Width and half-amplitude
    templim=np.nonzero(fit>np.max(fit/2))
    # print(np.shape(templim))
    width=(templim[0][-1]-templim[0][0])*10
    if (width>360):
        width=360

    # Preferred orientation from the fitted curve
    # print(new_r)
    id_temp=np.argmax(fit)
    pref=X_temp[id_temp]
    r_pref=new_r[id_temp]
    r_orth=np.min([new_r[0], new_r[-1]])
    osi=(r_pref-r_orth)/(r_pref+r_orth)
    return pref, osi, width, r_pref, r_orth