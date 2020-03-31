from tools.Data.Subscriber import Subscriber
from tools.Data.Publisher import Publisher
from tools.Data.Save import Save
from tools.Data.Load import Load
from tools.Fail_condition import Fail
from tools.Learning.Learning import Learning
from tools.supervisor import Supervisor
from tools.Controller.Robot import Robot
import rospy,sys,subprocess, pickle
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter


save = Save('data/')
load = Load('data/')
robot = Robot()
Sub = Subscriber()
s = [Sub.goal_1,Sub.goal_2,Sub.target_pose]
Pub = Publisher()
fail = Fail(s)

command = ("ls data/action | grep")

Num_goal = 2

def initialize():
    global state, action
    state , action = save.initDataframe(Num_goal)


def shutdown():
    print ('ros shutdown')
    
def main():
    global state, action
    dataNumber = 13
    Max_trajectory = 50
    noise = [0.0,0.0]
    sampling_flag = False
    save_flag = False
    fail_flag = False
   
    result = {'model_Num' :[]
             ,'noise_x':[]
             ,'noise_y':[]}
    row = 0
    col = 0
    initialize()
    rospy.init_node('Demo', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    for t in range(Max_trajectory):
            
        Sup_x = Supervisor(noise[0])
        Sup_y = Supervisor(noise[1])
        [a_x,E_x,IE_x] = [0.0,0.0,0.0]
        [a_y,E_y,IE_y] = [0.0,0.0,0.0]
        
        button = True
        k=0

        while True:
            s = [Sub.goal_1,Sub.goal_2,Sub.endeffector_pose]
            fail = Fail(s)
            
            a_x, a_y = robot.policy(s,k)
            axes = [a_y, a_x]
            a = axes
            temp_state, temp_action = save.tempDataframe(s, a, Num_goal)
            fail_flag = fail.fail_check(Sub.simulationTime)
            if button :
                Pub.reset()
                initialize()
                k += 1
                k %= 2
                sampling_flag = True
                button = False
            elif fail_flag or (Sub.simulationState == 0) :
                Pub.sim_stop()
                initialize()
                k += 1
                k %= 2
                sampling_flag = False
                button = True
                fail_flag = False
            
            if sampling_flag :
                state = save.dataAppend(state,temp_state)
                action = save.dataAppend(action,temp_action)
                action1 = Sup_y.sample_action(axes[0])
                action2 = Sup_x.sample_action(axes[1])
                sample_action = [action1,action2]
                Pub.actionInput(sample_action)
                if Sub.success == True :
                    save_flag = True

            if save_flag :
                print(Sub.success)
                Pub.sim_stop()
                save.dataSave(state,action,dataNumber)
                save_flag = False
                sampling_flag = False
                Sub.success = False
                button = True
                if (dataNumber)%2==0 :
                    dataNumber += 1
                    break
                dataNumber += 1
            
            rate.sleep()
        
        initialize()
        Num_data = int(subprocess.check_output(command + " action | wc -l", shell=True))
        for i in range(Num_data):
            _state, _action = load.dataLoad(i+1)
            state = save.dataAppend(state,_state)
            action = save.dataAppend(action,_action)
        N = state.shape[0]
        
        X = state
        Y = action

        # OMGP = Learning('OMGP',20,X,Y)
        # OMGP.learning()
        GMLRM = Learning('GMLRM',100,X,Y)
        GMLRM.learning()


        # noise = [model.model.Noise,model.model.Noise]
        noise = [GMLRM.model.Noise[0,0],GMLRM.model.Noise[1,1]]
        result['noise_x'].append(noise[0])
        result['noise_y'].append(noise[1])
        result['model_Num'].append(i+1)
        df = pd.DataFrame(result)
        df.to_excel('data/noise/noise.xlsx')

        print("="*40)
        print("Optimized Noise x: %f, Noise y: %f" %(noise[0],noise[1]))
        print(" \t model saved" )
        print(" \t Number of step %i " %(N))
        print("="*40)

    
    rospy.spin()


if __name__=='__main__':
    main()
