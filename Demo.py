from tools.Data_Subscriber import Subscriber
from tools.Data_Publisher import Publisher
from tools.Data_save import Save
from tools.Fail_condition import Fail
from tools.supervisor import Supervisor
import rospy

save = Save('data/')
Sub = Subscriber()
Pub = Publisher()
fail = Fail()
Sup = Supervisor(0)
Num_goal = 1

def initialize():
    global state, action
    state , action = save.initDataframe(Num_goal)


def shutdown():
    print ('ros shutdown')
    

def main():
    global state, action
    dataNumber = 0
    sampling_flag = False
    save_flag = False
    fail_flag = False

    initialize()
    rospy.init_node('Demo', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    
    while True:
        
        axes, buttons = Pub.joyInput()
        s = [Sub.goal_1,Sub.goal_2,Sub.endeffector_pose]
        a = axes
        temp_state, temp_action = save.tempDataframe(s, a, Num_goal)
        
        if buttons[2] :
            Pub.reset()
            initialize()
            sampling_flag = True

        elif buttons[1] :
            Pub.sim_stop()
            initialize()
            sampling_flag = False
        
        if Sub.success == True :
            save_flag = True
            
        
        if sampling_flag :
            state = save.dataAppend(state,temp_state)
            action = save.dataAppend(action,temp_action)
            Pub.actionInput(Sup.sample_action(axes))

        if save_flag :
            Pub.sim_stop()
            save.dataSave(state,action,dataNumber)
            dataNumber += 1
            save_flag = False
            sampling_flag = False
            Sub.success = False
        
        rate.sleep()
    rospy.spin()


if __name__=='__main__':
    main()
