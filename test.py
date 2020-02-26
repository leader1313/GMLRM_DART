from tools.Data_Subscriber import Subscriber
from tools.Data_Publisher import Publisher
from tools.Data_save import Save
from tools.Fail_condition import Fail
import pickle
import rospy, torch
import numpy as np


Sub = Subscriber()
Pub = Publisher()
fail = Fail()

#########  Load model  #######
GMM_filename_x = 'GMM_model/learner_x.pickle'
GMM_filename_y = 'GMM_model/learner_y.pickle'
GP_filename_x = 'GP_model/learner_x.pickle'
GP_filename_y = 'GP_model/learner_y.pickle'
with open(GMM_filename_x, 'rb') as f:
    GMM_x = pickle.load(f)
with open(GMM_filename_y, 'rb') as f:
    GMM_y = pickle.load(f)    
with open(GP_filename_x, 'rb') as f:
    GP_x = pickle.load(f)
with open(GP_filename_y, 'rb') as f:
    GP_y = pickle.load(f)    




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
        # s = [s1.x,s2.x,s1.y,s2.y,s3.x,s3.y]
        s = [s1.x,s1.y,s3.x,s3.y]
        temp_state = s
        np_temp = np.asarray(temp_state)[None,...]
        te_temp = torch.from_numpy(np_temp).float()
        mm_action_x, ss_action_x = GP_x.predict(te_temp)
        mm_action_y, ss_action_y = GP_y.predict(te_temp)
        
        
        a_x = mm_action_x
        a_y = mm_action_y
        GP_a = [a_y,a_x]
        
        k=0
        a_x = GMM_x.predict(s)[k]
        a_y = GMM_y.predict(s)[k]
        GMM_a = [a_y,a_x]
        
        print(s)
        print(GP_a)
        print(GMM_a)

        if kflag : 
            k = 0
            a = GP_a
        else : 
            k = 0
            a = GMM_a
        
        
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
