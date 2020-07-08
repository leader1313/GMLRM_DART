from tools.Data.Subscriber import Subscriber
from tools.Data.Publisher import Publisher
from tools.Data.Save import Save
from tools.Data.Load import Load
from tools.Data.Clear import clear
from tools.Fail_condition import Fail
from tools.Learning.Learning import Learning
from tools.supervisor import Supervisor
from tools.Controller.Robot import Robot
import rospy,sys,subprocess, pickle
import matplotlib.pyplot as plt
import pandas as pd
import xlsxwriter, os


save = Save('data/')
load = Load('data/')

Sub = Subscriber()
# s = [Sub.goal_1,Sub.goal_2,Sub.goal_3,Sub.target_pose]
s = [Sub.goal_1,Sub.goal_2,Sub.target_pose]
Pub = Publisher()
Num_goal = 2
fail = Fail(s)

command = ("ls data/action | grep")



def initialize():
    global state, action
    state , action = save.initDataframe(Num_goal)

def shutdown():
    print ('ros shutdown')
    
def main():
    global state, action
    ##clear whole file from data dir
    clear()
    ####### Initialize Parameters
    dataNumber = 1
    Max_trajectory = 10
    sampling_flag = False
    save_flag = False
    fail_flag = False
    robot = Robot(Num_goal)

    row = 0
    col = 0
    initialize()
    rospy.init_node('Demo', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    for t in range(Max_trajectory):
        Sup_x = Supervisor(0.0)
        Sup_y = Supervisor(0.0)
        [a_x,E_x,IE_x] = [0.0,0.0,0.0]
        [a_y,E_y,IE_y] = [0.0,0.0,0.0]
        
        button = True
        k=1

        while True:
            # s = [Sub.goal_1,Sub.goal_2,Sub.goal_3,Sub.endeffector_pose]
            s = [Sub.goal_1,Sub.goal_2,Sub.endeffector_pose]
        
            fail = Fail(s)
            a_x, a_y = robot.policy(s,k)
            
            axes = [a_y, a_x]
            a = axes
            temp_state, temp_action = save.tempDataframe(s, a, Num_goal)
            
            fail_flag = fail.fail_check(Sub.simulationTime)
            if button :
                robot = Robot(Num_goal)
                Pub.reset(t)
                initialize()
                
                sampling_flag = True
                button = False
            elif fail_flag or (Sub.simulationState == 0) :
                Pub.sim_stop()
                initialize()
                sampling_flag = False
                button = True
                fail_flag = False
            
            if sampling_flag :
                # temp_action['v_y1'], temp_action['v_x1']= action1, action2
                state = save.dataAppend(state,temp_state)
                action = save.dataAppend(action,temp_action)
                action1 = Sup_y.sample_action(axes[0])
                action2 = Sup_x.sample_action(axes[1])
                sample_action = [action1,action2]
                Pub.actionInput(sample_action)
                if Sub.simulationTime >1.0 :
                    fail.simple_success()
                
                if (Sub.success == True) or (fail.success==True) :
                    save_flag = True
            if save_flag :
                k += 1
                k %= Num_goal
                Pub.sim_stop()
                save.dataSave(state,action,dataNumber)
                save_flag = False
                sampling_flag = False
                Sub.success = False
                fail.success = False
                button = True
                if (dataNumber)%Num_goal==0 :
                    dataNumber += 1
                    break
                dataNumber += 1
            
            rate.sleep()
        
        if ((dataNumber-1) % 2 ==0) :
            initialize()
            Num_data = int(subprocess.check_output(command + " action | wc -l", shell=True))
            for i in range(Num_data):
                _state, _action = load.dataLoad(i+1)
                state = save.dataAppend(state,_state)
                action = save.dataAppend(action,_action)
            N = state.shape[0]
            
            X = state
            Y = action
            Y1 = Y['v_x1']
            Y2 = Y['v_y1']
            
            model = Learning('IMGP',30,X,Y)
            # model = Learning('HIMGP',30,X,Y)
            model.learning(int((dataNumber-1)/Num_goal))            
            
            print("="*40)
            print(" \t model saved" )
            print(" \t Number of step %i " %(N))
            print("="*40)

    
    rospy.spin()


if __name__=='__main__':
    main()
