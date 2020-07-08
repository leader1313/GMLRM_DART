#-*- coding:utf-8 -*-
from tools.Data.Load import Load
from tools.Data.Save import Save
from tools.Learning.Learning import Learning
import sys,subprocess
import pandas as pd
import numpy as np
import pickle
import joblib

save = Save('data/')
load = Load('data/')

command = ("ls data/action | grep")
Num_data = int(subprocess.check_output(command + " action | wc -l", shell=True))
Num_goal = 2

def main():
    state,action = save.initDataframe(Num_goal)
    filename = 'HIMGP_model/DART/learner8.pickle'
    model1 = joblib.load(filename)
    filename = 'HIMGP_model/DART/learner9.pickle'
    model2 = joblib.load(filename)

    # for i in range(Num_data):
    #     _state, _action = load.dataLoad(i+1)
    #     state = save.dataAppend(state,_state)
    #     action = save.dataAppend(action,_action)
    
    # N = state.shape[0]
    
    #==================== random initialize parameter =======================
    X = model2.X
    Y = model2.Y
    K = model1.N_K
    
    Learning('HIMGP',30,X,Y,K=K).learning(10)

if __name__=='__main__':
    main()