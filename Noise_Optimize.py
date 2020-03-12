#-*- coding:utf-8 -*-
from tools.Data_Subscriber import Subscriber
from tools.Data_Publisher import Publisher
from tools.Data_save import Save
from tools.Data_load import Load
from tools.GMLRM import GMLRM
from tools.Noise import Noise
import pickle
import rospy, torch
import numpy as np
import sys,subprocess
import pandas as pd


save = Save('data/')
load = Load('data/')

command = ("ls data/action | grep")
Num_data = int(subprocess.check_output(command + " action | wc -l", shell=True))
Num_goal = 2

state,action = save.initDataframe(Num_goal)


#########  Load model  #######
GMM_filename = 'GMM_model/learner.pickle'
with open(GMM_filename, 'rb') as f:
    GMM = pickle.load(f)

state = GMM.X
action = GMM.Y


pre_noise = np.array([0.00168622,0.00168622])

noise = Noise(state,action,GMM,pre_noise)
noise.optimize()
