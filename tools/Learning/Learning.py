#-*- coding:utf-8 -*-
from tools.Data.Load import Load
from tools.Learning.GMLRM import GMLRM
from tools.Learning.GP import GPRegression
from tools.Learning.OMGP import OMGP
from tools.Learning.kernel import GaussianKernel
from sklearn.externals import joblib
import sys,subprocess
import pandas as pd
import numpy as np
import pickle



load = Load('data/')

class Learning:
    def __init__(self, method, max_iter, train_x, train_y):
        self.X = load.dataframe_to_numpy(train_x)
        self.Y = load.dataframe_to_numpy(train_y)
        self.T = max_iter
        self.K = 2                                       # solution 수
        self.M = 3                                    # Number of model
        self.Noise = np.zeros(self.Y.shape[1])
        if method == 'GMLRM':
            self.model = GMLRM(self.X,self.Y,self.K,self.M,self.T)
            
        elif method == 'OMGP' :
            self.X = load.num_to_ten(self.X).float()
            self.Y = load.num_to_ten(self.Y).float()
            self.model = OMGP(self.X, self.Y, self.K, self.T, GaussianKernel)
            
        self.md_name = self.model.__class__.__name__

    def learning(self):
        self.model.learning()
        self.Noise = self.model.Noise
        #save model as Pickle
        file_name = self.md_name+'_model/learner.pickle' 
        joblib.dump(self.model, file_name)
        print('='*56)
        print('+'*14+ self.md_name +' model Saved!!'+'+'*14)
        print('='*56)
        