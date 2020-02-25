from tools.Data_Subscriber import Subscriber
from tools.Data_Publisher import Publisher
from tools.Data_save import Save
from tools.Fail_condition import Fail
import pickle
import rospy


Sub = Subscriber()
Pub = Publisher()
fail = Fail()

#########  Load model  #######
filename_x = 'model/learner_x.pickle'
filename_y = 'model/learner_y.pickle'
with open(filename_x, 'rb') as f:
    Ler_x = pickle.load(f)
with open(filename_y, 'rb') as f:
    Ler_y = pickle.load(f)    

model_x = Ler_x['model']
model_y = Ler_y['model']


def shutdown():
    print ('ros shutdown')
    
def main():
    global state, action
    dataNumber = 0
    sampling_flag = False
    save_flag = False
    fail_flag = False
    kflag = False
    rospy.init_node('Test', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    
    while True:
        
        axes, buttons = Pub.joyInput()
        
        s1 = Sub.goal_1
        s2 = Sub.goal_2
        s3 = Sub.endeffector_pose
        s = [s1.x,s2.x,s1.y,s2.y,s3.x,s3.y]
        if kflag : k = 1
        else : k = 0
        a_x = model_x.predict(s)[k]
        a_y = model_y.predict(s)[k]
        a = [a_y,a_x]
        print(s)
        print(a)
        
        if buttons[2] :
            Pub.reset()
            sampling_flag = True

        elif buttons[1] :
            Pub.sim_stop()
            sampling_flag = False
        
        elif buttons[0] :
            kflag = not kflag
            print(kflag)

        if Sub.success == True :
            save_flag = True
            
        
        if sampling_flag :
            Pub.actionInput(a)
            
        if save_flag :
            Pub.sim_stop()
            save_flag = False
            sampling_flag = False
            Sub.success = False
        
        rate.sleep()
    rospy.spin()


if __name__=='__main__':
    main()
