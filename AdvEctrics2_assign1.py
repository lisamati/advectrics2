#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
advectrics_A11.py

Purpose:
    Assignment 1, exercise 5. 
    Programme and run a Monte Carlo experiment on the e
ect
    of ignoring the uncertainty surrounding the estimation of the propensity
    score on the size of the test for no treatment e
ect when using strati
cation
    based on the propensity score.
    
Version:
    1       First start, generate data, treatment assignment 
    2       model propensity scores via logistic regression

Date:
    2020/01/13

Author:
    Lisa Timm
    Diego Dabed 
"""
###########################################################
### Imports
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt

###########################################################
def fillX(iN):
    """
    Purpose:
        Simulate potential outcomes drawn from a standard normal distribution

    Inputs:
        iN      number of observations

    Return value:
        mY     matrix with a column for potential outcomes for treated and non-treated individuals
    """
    vX1 = np.zeros(iN)
    vX0 = np.zeros(iN)
        for j in range(iN):
            vX1[j] = np.random.normal(0,1)
            vX0[j] = np.random.normal(0,1)
    return (vX1,vX0)

###########################################################
def treatassign(iN, vX1, vX0, iSigq):
    """
    Purpose:
        define a latent variable for treatment assignment    
        probability of treatment  =covariate 0 error
    
    Inputs:
        iN      number of observations
        vX1     vector of potential outcomes is treated 
        vX0     vector of potential outcomes if not treated 
        iSigq   variance of error term in latent variable 
        
        
    Return value:
        vD     treatment 
    """
    vTheta = np.zeros(iN)
    for j in range(iN):
        vTheta[j] = np.random.normal(0,iSigq)
            
    pstar = vX1 + vX0+ vTheta
    np.mean(pstar)
    
    vD = np.where(pstar>0, 0,1)
    np.mean(vD)

    return (vD)
##########################################################
def propensityscore(vD, vX1, vX0):
    """
    Purpose: 
        Setup a logit model to calculate the propensity score based on the simulated covariates 
        
    Inputs: 
        vD      vector with treatment indicator
        vX1     vector with covariates for potentially treated
        vX0     vector with covariates for potentially untreated
        
    Output
        vPx     propensity score, based on covariates    
    """
    vX = np.vstack((vX1,vX0)).T

    model=LogisticRegression(solver='lbfgs').fit(vX,vD)
    vPx=LogisticRegression.predict_proba(model,vX)[:,1]
    
    return(vPx)
     
##########################################################
def stratify(vPx, vX, )    
    """
    Purpose: 
        stratify sample with propensity score, assign an indicator for each stratum
        i.e. 1 for 0-0.1, 2 for 0.1-0.2 etc.
        
    Inputs: 
        vPx     vector with propensity scores
        mX      matrix with potential outcomes 
         
    Output
        vPx     propensity score, based on covariates    
    """
    data = {[vX1], 'x0': [vX0], 'treatment': [vD], 'propensity score': [vPx]}
    data=np.vstack((vX1,vX0, vD, vPx)).T
    df = pd.DataFrame(data, columns=['x1','x0', 'treatment','propensity score'])
    
    df['d1'] = np.where(vPx <= 0.1,1,0)
    df['d2'] = np.where(vPx <= 0.2,1,0) & np.where(vPx > 0.1,1,0)
    df['d3'] = np.where(vPx <= 0.3,1,0) & np.where(vPx > 0.2,1,0)
    df['d4'] = np.where(vPx <= 0.4,1,0) & np.where(vPx > 0.3,1,0)
    df['d5'] = np.where(vPx <= 0.5,1,0) & np.where(vPx > 0.4,1,0)
    df['d6'] = np.where(vPx <= 0.6,1,0) & np.where(vPx > 0.5,1,0)
    df['d7'] = np.where(vPx <= 0.7,1,0) & np.where(vPx > 0.6,1,0)
    df['d8'] = np.where(vPx <= 0.8,1,0) & np.where(vPx > 0.7,1,0)
    df['d9'] = np.where(vPx <= 0.9,1,0) & np.where(vPx > 0.8,1,0)
    df['d10'] = np.where(vPx <= 1,1,0) & np.where(vPx > 0.9,1,0)

    df.head()
    
    df.mean()

##########################################################
### main
def main():
    # Magic numbers
    set seed 123
    iN = 1000
    iSigq = 5

    # Initialisation
    fillX(iN)
    treatassign(iN, vX1, vX0, iSigq)
    
    # Estimation
    propensityscore(vD, vX1, vX0)

    # Output
    print ("output")

###########################################################
### start main
if __name__ == "__main__":
    main()
