import pandas as pd
from pandas import Series, DataFrame

class Load(object):
    def __init__(self,dir_path):
        self.dir = dir_path

    def load_excel(self, excel_name):
        excel_name = excel_name +'.xlsx'
        dataframe = pd.read_excel(self.dir + excel_name, index_col=0)
        return dataframe

    def dataframe_to_numpy(self, dataframe):
        numpy_array = dataframe.to_numpy()
        return numpy_array
    
    def dataLoad(self, dataNumber):
        state_name = 'state/state'+ str(dataNumber)
        action_name = 'action/action'+ str(dataNumber)
        state = self.load_excel(state_name)
        action = self.load_excel(action_name)
        
        print('='*56)
        print ('+'*18+' Data number '+ str(dataNumber) + ' loaded'+'+'*18)
        print('='*56)
        return state, action
    



def shutdown():
    print ('ros shutdown')
    
def main():
    # rospy.init_node('Load', anonymous=True, disable_signals=True)
    # rospy.on_shutdown(shutdown)
    # rate = rospy.Rate(10)
    load = Load('data/')
    state2, action2 = load.dataLoad(2)
    state1,action1 = load.dataLoad(1)
    state = load.dataAppend(state1,state2)
    print(state1.shape)
    print(state2.shape)
    print(state.shape)

if __name__=='__main__':
    main()

