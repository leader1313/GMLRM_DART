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
    for i in range(Num_data):
        _state, _action = load.dataLoad(i+1)
        state = save.dataAppend(state,_state)
        action = save.dataAppend(action,_action)
    
    N = state.shape[0]
    
    
    #==================== random initialize parameter =======================
    X = state
    Y = action
    
    Learning('OMGP',10,X,Y).learning(12)

if __name__=='__main__':
    main()