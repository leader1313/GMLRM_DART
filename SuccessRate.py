from tools.Data.Subscriber import Subscriber
from tools.Data.Publisher import Publisher
from tools.Data.Save import Save
from tools.Fail_condition import Fail
import pickle, sys
import rospy, torch
import numpy as np
import pandas as pd
import joblib

#########  initialization   #######
Sub = Subscriber()
Pub = Publisher()


def shutdown():
    print ('ros shutdown')

def main():
    rospy.init_node('Success_rate', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    Max_trial = 200
    df = pd.DataFrame(columns=['Model_name', 'success_rate','var_s','var_f'] )

    #########  Load model       #######
    BC1_filename = 'OMGP_model/BC/learnerBC1.pickle'
    BC2_filename = 'OMGP_model/BC/learnerBC2.pickle'
    BC3_filename = 'OMGP_model/BC/learnerBC3.pickle'
    BC4_filename = 'OMGP_model/BC/learnerBC4.pickle'
    BC5_filename = 'OMGP_model/BC/learnerBC5.pickle'
    BC6_filename = 'OMGP_model/BC/learnerBC6.pickle'
    DART1_filename = 'OMGP_model/DART/learner5.pickle'
    DART2_filename = 'OMGP_model/DART/learner10.pickle'
    DART3_filename = 'OMGP_model/DART/learner15.pickle'
    DART4_filename = 'OMGP_model/DART/learner20.pickle'
    DART5_filename = 'OMGP_model/DART/learner25.pickle'
    DART6_filename = 'OMGP_model/DART/learner30.pickle'

    models = [DART1_filename,DART2_filename,DART3_filename,DART4_filename,DART5_filename,DART6_filename
                ,BC1_filename,BC2_filename,BC3_filename,BC4_filename,BC5_filename,BC6_filename]

    #########  model selection  #######
    for filename in models:
        model = joblib.load(filename)
        #mean of Variance
        Sum_ss = 0
        Sum_fs = 0
        N_s = 0
        N_f = 0
        var_s=0
        var_f=0
        successRate = 0
        trial = 0
        #########  100s trial       #######
        while (trial <= Max_trial) :
            trial += 1
            step=0
            temp_ss = 0
            finish_flag = False
            sampling_flag = False
            Sub.success = False
            fail_flag = False
            k = 0
            if trial > int((Max_trial)/2): k = 1
        #########  While trial      ####### 
            start_flag = True   
            
            while True:        
            #### State        
                s = [Sub.goal_1,Sub.goal_2,Sub.endeffector_pose]
                #### fail check
                fail = Fail(s)
                fail_flag = fail.fail_check(Sub.simulationTime)

                s = [s[0].x,s[1].x,s[0].y,s[1].y,s[2].x,s[2].y]

                temp_state = s
                np_temp = np.asarray(temp_state)[None,...]
                te_temp = torch.from_numpy(np_temp).float()
                mm_action, ss_action = model.predict(te_temp)

            #### Action and regret
                a_x = mm_action[k][0][0]
                a_y = mm_action[k][0][1]
                ss  = ss_action[k]
                # Constraint for safe
                if abs(a_y) > 0.9 :
                    a_y /= abs(a_y)
                    a_y *= 0.9
                OMGP_a = [a_y,a_x]   
                a = OMGP_a
                
            #### Control
                #### Start
                if start_flag :
                    Pub.reset()
                    sampling_flag = True
                    fail_flag =False
                    start_flag = False
                #### Finish
                if (Sub.success == True) or (fail_flag == True) :
                    finish_flag = True
                elif (Sub.simulationState == 0) :
                    Pub.sim_stop()
                    trial -= 1
                    break

                if (Sub.simulationTime > 1.0) and (sampling_flag) :
                    step += 1
                    temp_ss += ss
                    Pub.actionInput(a)

                if finish_flag :
                    Pub.sim_stop()
                    temp_ss /= step
                    if Sub.success == True : 
                        N_s += 1
                        Sum_ss += temp_ss
                    elif fail_flag == True : 
                        N_f += 1
                        Sum_fs += temp_ss
                    break
                    
                rate.sleep()
            print('Model : %s , Trial : %i, s : %i, f : %i'%(filename,trial+1, N_s,N_f))
        successRate = N_s/Max_trial
        if N_s > 0:
            var_s = Sum_ss/N_s
        else : var_s = 0
        if N_f > 0:
            var_f = Sum_fs/N_f
        else : var_f = 0
        
        df2 = pd.DataFrame({"Model_name": filename, 
                        "success_rate":successRate,
                        "var_s":var_s,
                        "var_f":var_f}) 
        df = df.append(df2, ignore_index=True)
        
        print(df)
        df.to_excel('data/success_rate/result.xlsx', sheet_name = 'new_sheet_name')
    


if __name__=='__main__':
    main()
