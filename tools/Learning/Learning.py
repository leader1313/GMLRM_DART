#-*- coding:utf-8 -*-
from tools.Data.Load import Load
from tools.Learning.GMLRM import GMLRM
from tools.Learning.GP import GPRegression
from tools.Learning.OMGP import OMGP
from tools.Learning.kernel import GaussianKernel
from tools.Learning.gp_regressors import GPRegressor
from tools.Learning.nonparametric_gp_regression  import nonparametric_gp_regression
from tools.Learning.utils.constraints_generator import constraints_generator
import sys,subprocess
import pandas as pd
import numpy as np
import networkx as nx
import pickle, joblib



load = Load('data/')

class Learning:
    def __init__(self, method, max_iter, train_x, train_y):
        self.X = load.dataframe_to_numpy(train_x)
        self.Y = load.dataframe_to_numpy(train_y)
        self.T = max_iter
        self.K = 2                                      # solution 수
        self.M = 3   
        self.method = method                                 # Number of model
        if method == 'IMGP':
            kernel_name   = 'gaussian'
            kernel_params = np.array([1.0, 0.3])
            self.kernel = [kernel_name, kernel_params]
            self.inputs  = self.X
            self.targets = self.Y
            var = 0.1
            self.beta   = 1.0/var 
            self.alpha  = 1.0
            self.iters  = self.T
            self.K = 5
            self.prob_thresh = 0.001
        
        else :
            if method == 'GMLRM':
                self.model = GMLRM(self.X,self.Y,self.K,self.M,self.T)

            
            
            elif method == 'OMGP' :
                self.X = load.num_to_ten(self.X).float()
                self.Y = load.num_to_ten(self.Y).float()
                self.model = OMGP(self.X, self.Y, self.K, self.T, GaussianKernel)
                
            self.md_name = self.model.__class__.__name__

    def learning(self,i):
        if self.method =='IMGP':
            self.model, z_disc_R, lower_bound = nonparametric_gp_regression(self.inputs, self.targets,
                                        self.kernel, self.beta, self.alpha, self.iters, self.K, self.prob_thresh, repeats=1)
            print('Number of GP Mixture : ',len(self.model))
        else: 
            self.model.learning()
        
        #save model as Pickle
        file_name = self.method+ '_model/learner'+ str(i) +'.pickle'
        joblib.dump(self.model, file_name)
        print('='*56)
        print('+'*14+ self.method +' model Saved!!'+'+'*14)
        print('='*56)
        