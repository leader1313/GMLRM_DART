import pandas as pd
from pandas import Series, DataFrame

def load_excel( data_path, data_name, episode_Num):
    data_name = data_name + str(episode_Num)+'.xlsx'
    dataframe = pd.read_excel(data_path + data_name,index_col=0)
    return dataframe

def dataframe_to_numpy(dataframe):
    numpy_array = dataframe.to_numpy()
    return numpy_array

def numpy_to_tensor( numpy_array):
    tensor = torch.from_numpy(numpy_array)
    return tensor

def transform( data, data_path, data_name, episode_Num):
    data = load_excel(data_path, data_name, episode_Num)
    data = dataframe_to_numpy(data)
    data = numpy_to_tensor(data)
    return data

def data_slice(data, num_of_data):
    data = data[:, 1:num_of_data+1]
    return data