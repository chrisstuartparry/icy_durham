# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:17:38 2025

@author: qxp518
"""

import numpy as np

from numpy import arange, concatenate, zeros, linspace, floor, array, pi
from numpy import sin, cos, sqrt, random, histogram, abs, sqrt, max
import time as ts
import matplotlib.pyplot as plt # Matplotlib plotting library
import scipy as sp

import epic1d as sim
import data as da
from automated_sim import cell_array
import pickle




def curven(data,*params):
    ''' fit curve of power n '''
    A= params[0]
    B = params[1]
    n = params[2]
    y = (A*(data**n)) + B
    return y


def fit(xdata,ydata,function,start_params):
    ''' a fitting function that fits the defined "function" to the x and y data
    and using the start params as the initial guess, requires scypy'''
    fitted_params, covariance =sp.optimize.curve_fit(function, xdata, ydata, p0=start_params)
    
    
    return fitted_params,covariance

def bestfit(xdata,ydata,function,start_params):
    fitted_param,covariance = fit(xdata,ydata,function,start_params)
    y = function(xdata,*fitted_param)
    return y, fitted_param
    

# chris woz ere

def supconduct(xdata,ydata,cuttoff,*params):
    Aback = params[0]
    Bback = params[1]
    Asup = params[2]
    Bsup = params[3]
    nsup = params[4]
    # split data before and after superconductive failiure
    back_xdata = xdata[:,cuttoff]
    back_ydata = ydata[:,cuttoff]    # #   Get equivalent y
    super_data = xdata[cuttoff,:]
    super_ydata = ydata[cuttoff,:]
    back_params = [Aback,Bback,1]
    background_y, background_params = bestfit(xdata,ydata,curven,back_params)  
    # supbtract background behavour
    afterdata = super_ydata - curven(super_ydata, back_params)
    
    # fit superconductor
    super_params = [Asup,Bsup,nsup]
    true_super_y, super_fit_param = bestfit(super_data, afterdata, curven, super_params)
    return super_data,true_super_y,super_fit_param
    
    
    
    
    
    
    




