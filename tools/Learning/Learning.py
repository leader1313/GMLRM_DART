#-*- coding:utf-8 -*-
from tools.Data.Load import Load
from tools.Learning.GMLRM import GMLRM
from tools.Learning.GP import GPRegression
from tools.Learning.OMGP import OMGP
from tools.Learning.IMGP import IMGP
from tools.Learning.HIMGP import HIMGP
from tools.Learning.kernel import GaussianKernel
import sys,subprocess
import pandas as pd
import numpy as np
import pickle, joblib



load = Load('data/')

class Learning:
    def __init__(self, method, max_iter, train_x, train_y,DART=None,old_sigma = [], K = []):
        # self.X = load.dataframe_to_numpy(train_x)
        # self.Y = load.dataframe_to_numpy(train_y)
        self.T = max_iter
        self.K = 5                                   # solution 수
        self.Mixture = 3   
        self.method = method                                 # Number of model
       
        if method == 'GMLRM':
            self.model = GMLRM(self.X,self.Y,self.K,self.M,self.T)

        elif method == 'OMGP' :
            self.X = load.num_to_ten(self.X).float()
            self.Y = load.num_to_ten(self.Y).float()
            self.model = OMGP(self.X, self.Y, 2, self.T, GaussianKernel)

        elif method == 'IMGP' :
            self.X = load.num_to_ten(self.X).float()
            self.Y = load.num_to_ten(self.Y).float()
            self.model = IMGP(self.X, self.Y, self.K, self.T, GaussianKernel)

        elif method == 'HIMGP' :
            # self.X = load.num_to_ten(self.X).float()
            # self.Y = load.num_to_ten(self.Y).float()
            self.X = train_x
            self.Y = train_y
            self.model = HIMGP(self.X, self.Y, self.K, self.T, GaussianKernel,old_sigma = old_sigma, K = K)
            
        self.md_name = self.model.__class__.__name__

    def learning(self,i):
        self.model.learning()
        
        #save model as Pickle
        file_name = self.method+ '_model/learner'+ str(i) +'.pickle'
        joblib.dump(self.model, file_name)
        print('='*56)
        print('+'*14+ self.method +' model Saved!!'+'+'*14)
        print('='*56)
        