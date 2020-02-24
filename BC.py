import pandas as pd
from pandas import Series, DataFrame
from tools.Data_process import *

action_path = 'data/action/'
action_name = 'action'
dataNum = 1

df = load_excel(action_path,action_name,dataNum)
np = dataframe_to_numpy(df)

print(np[:,0])