from tools.Data.Subscriber import Subscriber
from tools.Data.Publisher import Publisher
from tools.Data.Save import Save
from tools.Fail_condition import Fail
import pickle
import rospy, torch
import numpy as np
import joblib, random


Sub = Subscriber()
Pub = Publisher()

#########  Load model  #######
# GMLRM_filename = 'GMLRM_model/learner.pickle'
# OMGP_filename1 = 'OMGP_model/learner11x.pickle'
# IMGP_filename = 'IMGP_model/DART/learner12.pickle'
# IMGP_filename = 'IMGP_model/push_pick_BC/learner8.pickle'
# IMGP_filename = 'IMGP_model/push_pick_DART/learner4.pickle'
# IMGP_filename = 'IMGP_model/learner12.pickle'
IMGP_filename = 'OMGP_model/learner7.pickle'
# IMGP_filename = 'OMGP_model/limited_random_DART/learner12.pickle'
# IMGP_filename = 'OMGP_model/push_pick_DART/learner12.pickle'
# IMGP_filename = 'OMGP_model/learner12.pickle'

# IMGP_filename = 'IMGP_model/DART/learner30.pickle'

# GMM = joblib.load(GMLRM_filename)
# OMGP = joblib.load(OMGP_filename)
IMGP = joblib.load(IMGP_filename)
# OMGPx = joblib.load(OMGP_filename1)
# OMGPy = joblib.load(OMGP_filename2)

def shutdown():
    print ('ros shutdown')
    
def main():
    global state, action

    sampling_flag = False
    save_flag = False
    fail_flag = False
    kflag = False
    rospy.init_node('Test', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    k=0
    temp_ss = 0
    t=0
    n_regressors = IMGP.M
    print(n_regressors)
    while True:
        
        axes, buttons = Pub.joyInput()
        
        s1 = Sub.goal_1
        s2 = Sub.goal_2
        s3 = Sub.endeffector_pose
        s = [s1.x,s2.x,s1.y,s2.y,s3.x,s3.y]
        # s = [s1.x,s1.y,s3.x,s3.y]
        # s = [s1.y,s2.y,s3.y]
        temp_state = s
        np_temp = np.asarray(temp_state)[None,...]
        te_temp = torch.from_numpy(np_temp).float()

        mm_action, ss_action = IMGP.predict(te_temp)
        # mm_actionx, ss_actionx = OMGPx.predict(te_temp)
        # mm_actiony, ss_actiony = OMGPy.predict(te_temp)
        

        # a_x = mm_action[0][0][0]
        # a_y = mm_action[0][0][1]
        # a_x = mm_action[k][0][0]
        # a_y = mm_action[k][0][1]
        # a_x = mm_actionx[k][0]
        # a_y = mm_actiony[k][0]
        
        # OMGP_a = [a_y,a_x]
        
        # a_x = 0.0
        # a_y = mm_action[k][0]
        # OMGP_a = [a_y,a_x]
        # if ss_action[0] > ss_action[1]:
        #     k = 1
        # else: k = 0
        # ss, k = ss_action.max(0)
        # k = int(k)
        print(k)
        a_x = mm_action[k][0][0]
        a_y = mm_action[k][0][1]
        IMGP_a = [a_y,a_x]
       
        # a_x = GMM.predict(s)[k][0]
        # a_y = GMM.predict(s)[k][1]
        # GMM_a = [a_y,a_x]
        
        # [means, vars] = IMGP[k].get_target_predictions(np_temp)
    
        # a_x = means[0][0]
        # a_y = means[0][1]
        # if abs(a_y) >0.9 :
        #     a_y /= abs(a_y)
        #     a_y *= 0.9
        # IMGP_a = [a_y,a_x]
        
       
        a = IMGP_a

        if buttons[2] :
            time = int(random.uniform(-1,12))
            Pub.reset(time)
            sampling_flag = True

        elif buttons[1] :
            Pub.sim_stop()
            temp_ss /= t
            print(temp_ss)
            sampling_flag = False
        
        elif (buttons[0]) :
            # kflag = not kflag
            k += 1
            k %=n_regressors
            print(kflag)

        if Sub.success == True :
            save_flag = True
            
        
        if sampling_flag :
            t+=1
            temp_ss += ss_action
            Pub.actionInput(a)
            
        if save_flag :
            Pub.sim_stop()
            temp_ss /= t
            print(temp_ss)
            temp_ss = 0
            save_flag = False
            sampling_flag = False
            Sub.success = False
        
        rate.sleep()
    rospy.spin()


if __name__=='__main__':
    main()
