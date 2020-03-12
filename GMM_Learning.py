#-*- coding:utf-8 -*-
from tools.Data_load import Load
from tools.Data_save import Save
from tools.GMLRM import GMLRM
import sys,subprocess
import pandas as pd
import numpy as np
import pickle

save = Save('data/')
load = Load('data/')

command = ("ls data/action | grep")
Num_data = int(subprocess.check_output(command + " action | wc -l", shell=True))
Num_goal = 2

def main():
    state,action = save.initDataframe(Num_goal)
    for i in range(Num_data):
        _state, _action = load.dataLoad(i+1)
        state = save.dataAppend(state,_state)
        action = save.dataAppend(action,_action)
    N = state.shape[0]
    
    state = load.dataframe_to_numpy(state)
    action = load.dataframe_to_numpy(action)

    
    #==================== random initialize parameter =======================
    X = state
    Y = action

    T = 300
    K = 2                                       # solution 수
    M = 2                                    # Number of model
    GM = GMLRM(X,Y,K,M,T)

    GM.EM()

    learner = GM

    print("="*40)
    with open('GMM_model/learner'+'.pickle', 'wb') as handle:
        pickle.dump(learner, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Optimized Noise x: %f, Noise y: %f" %(GM.var[0,0],GM.var[1,1]))
    print(" \t model saved" )
    print(" \t Number of step %i " %(N))
    print("="*40)

if __name__=='__main__':
    main()