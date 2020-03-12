#-*- coding:utf-8 -*-
from tools.Data_load import Load
from tools.Data_save import Save
from tools.kernel import GaussianKernel
from tools.GP import GPRegression
from sklearn.externals import joblib
import sys,subprocess
import pandas as pd
import pickle, torch

save = Save('data/')
load = Load('data/')

command = ("ls data/action | grep")
Num_data = int(subprocess.check_output(command + " action | wc -l", shell=True))
Num_goal = 2

def main():
    state,action = save.initDataframe(Num_goal)
    for i in range(Num_data):
        _state, _action = load.dataLoad(i)
        state = save.dataAppend(state,_state)
        action = save.dataAppend(action,_action)
    N = state.shape[0]
    state = load.dataframe_to_numpy(state)
    action = load.dataframe_to_numpy(action)
    action_x = action[:,0][...,None]
    action_y = action[:,1][...,None]
    state = load.num_to_ten(state)
    action_x = load.num_to_ten(action_x)
    action_y = load.num_to_ten(action_y)
    
    #==================== random initialize parameter =======================
    X = state.float()
    Y1 = action_x.float()
    Y2 = action_y.float()
    
    kern1 = GaussianKernel()
    kern2 = GaussianKernel()

    model1 = GPRegression(X, Y1, kern1)
    model2 = GPRegression(X, Y2, kern2)

    print("params", torch.exp(model1.kern.param()[0]), torch.exp(model1.sigma), model1.negative_log_likelihood())
    print("params", torch.exp(model2.kern.param()[0]), torch.exp(model2.sigma), model2.negative_log_likelihood())
    model1.learning()
    model2.learning()
    print("params", torch.exp(model1.kern.param()[0]), torch.exp(model1.sigma), model1.negative_log_likelihood())
    print("params", torch.exp(model2.kern.param()[0]), torch.exp(model2.sigma), model2.negative_log_likelihood())

    #save model as Pickle
    file_name_m1 = 'GP_model/learner_x.pickle' 
    file_name_m2 = 'GP_model/learner_y.pickle' 
    joblib.dump(model1, file_name_m1)
    joblib.dump(model2, file_name_m2)
    print('='*56)
    print('+'*14+' model Saved!!'+'+'*14)
    print('='*56)

if __name__=='__main__':
    main()