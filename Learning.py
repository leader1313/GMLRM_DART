#-*- coding:utf-8 -*-
from tools.Data_load import Load
from tools.Data_save import Save
from tools.GMLRM import GMLRM
import sys,subprocess
import pandas as pd
import pickle

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
    
    #==================== random initialize parameter =======================
    X = state
    Y1 = action_x
    Y2 = action_y
    
    T = 100
    K = 2                                       # solution 수
    M = 100                                    # Number of model
    GM1 = GMLRM(X,Y1,K,M,T)
    GM2 = GMLRM(X,Y2,K,M,T)
    GM1.EM()
    GM2.EM()
    learner_y = {'model': GM1}
    learner_x = {'model': GM2}
    print("="*40)
    with open('model/learner_x'+'.pickle', 'wb') as handle:
        pickle.dump(learner_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('model/learner_y'+'.pickle', 'wb') as handle:
        pickle.dump(learner_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(" \t model saved" )
    print(" \t Number of step %i " %(N))
    print("="*40)

if __name__=='__main__':
    main()