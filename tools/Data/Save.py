import pandas as pd
from pandas import Series, DataFrame, concat

class Save(object):
    def __init__(self,file_path):
        self.file_path = file_path

    def initDataframe(self, Num_goal):
        if Num_goal == 0 : 
            self.init_state = DataFrame(columns =['goal_y1','goal_y2','y_e'])
            self.init_action = DataFrame(columns =['v_y1'])
        else:
            End = DataFrame(columns =['x_e','y_e'])
            Goal = DataFrame()
            for i in range(Num_goal):
                New_Goal = DataFrame(columns =['goal_x'+str(i+1),'goal_y'+str(i+1)])
                Goal = Goal.join(New_Goal)
            self.init_state = Goal.join(End)
            
            self.init_action = DataFrame(columns =['v_x1','v_y1'])

        return self.init_state, self.init_action

    def tempDataframe(self, state, action, Num_goal):
        if Num_goal == 0 : 
            self.temp_state = DataFrame({
                "goal_y1":[state[0].y],"goal_y2":[state[1].y],
                "y_e":[state[2].y]
                })
            self.temp_action = DataFrame({"v_y1":[action[0]]})
        else: 
            End = DataFrame({"x_e":[state[Num_goal].x],"y_e":[state[Num_goal].y]})
            
            Goal = DataFrame()
            for i in range(Num_goal):
                New_Goal = DataFrame({"goal_x"+str(i+1):[state[i].x],"goal_y"+str(i+1):[state[i].y]})
                Goal = concat([Goal,New_Goal],axis = 1)
            
            self.temp_state = Goal.join(End)
            self.temp_action = DataFrame({"v_x1":[action[1]],"v_y1":[action[0]]})

        

        return self.temp_state, self.temp_action

    def dataAppend(self, dataframe, temp_data):
        dataframe = dataframe.append(temp_data, ignore_index = True)
        return dataframe

    
    def dataSave(self, state, action, dataNumber):
        state = state.drop([0],axis=0)
        action = action.drop([0],axis=0)
    
        state.to_excel(self.file_path 
                + 'state/state'+ str(dataNumber) 
                + '.xlsx', sheet_name = 'new_sheet_name')
        action.to_excel(self.file_path 
                + 'action/action'+ str(dataNumber) 
                + '.xlsx', sheet_name = 'new_sheet_name')                
        print('='*56)
        print ('+'*18+' Data number '+ str(dataNumber) + ' saved'+'+'*18)
        print('='*56)


def shutdown():
    print ('ros shutdown')
    
def main():
    from Data_Subscriber import Subscriber
    from Data_Publisher import Publisher
    import rospy
    rospy.init_node('Data', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    save = Save('')
    state, action = save.initDataframe(2)
    
    Sub = Subscriber()
    Pub = Publisher()
    step = 0
    while True:
        step += 1
        axes, buttons = Pub.joyInput()
        s = [Sub.goal_1,Sub.goal_2,Sub.endeffector_pose]
        a = axes
        temp_state, temp_action = save.tempDataframe(s, a)
        action = save.dataAppend(action,temp_action)
        if step == 100 :
            save.dataSave(action,1,2)
            break
        rate.sleep()
    rospy.spin()


if __name__=='__main__':
    main()
