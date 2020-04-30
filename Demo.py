from tools.Data.Subscriber import Subscriber
from tools.Data.Publisher import Publisher
from tools.Data.Save import Save
from tools.Data.Clear import clear
from tools.Fail_condition import Fail
from tools.supervisor import Supervisor
import rospy

save = Save('data/')
Sub = Subscriber()
Pub = Publisher()
noise = 0.1980
Sup_x = Supervisor(noise)
Sup_y = Supervisor(noise)
Num_goal = 2

def initialize():
    global state, action
    state , action = save.initDataframe(Num_goal)


def shutdown():
    print ('ros shutdown')
    

def main():
    global state, action
    init_data_num = 3
    dataNumber = init_data_num
    sampling_flag = False
    save_flag = False
    fail_flag = False

    initialize()
    rospy.init_node('Demo', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    # clear()
    while True:
        axes, buttons = Pub.joyInput()
        
        s = [Sub.goal_1,Sub.goal_2,Sub.endeffector_pose]
        a = axes
        temp_state, temp_action = save.tempDataframe(s, a, Num_goal)
        
        if buttons[2] :
            Pub.reset(1)
            initialize()
            sampling_flag = True

        elif buttons[1] :
            Pub.sim_stop()
            initialize()
            sampling_flag = False

        elif buttons[0] :
            clear()
            dataNumber = init_data_num
            Pub.reset(1)
            initialize()
            sampling_flag = True

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
            dataNumber += 1
            save_flag = False
            sampling_flag = False
            Sub.success = False
        
        rate.sleep()
    rospy.spin()


if __name__=='__main__':
    main()
