from tools.Data.Subscriber import Subscriber
from tools.Data.Publisher import Publisher
from tools.Data.Save import Save
from tools.Fail_condition import Fail
import pickle, sys
import rospy, torch
import numpy as np
import pandas as pd
import joblib, random

#########  initialization   #######
Sub = Subscriber()
Pub = Publisher()


def shutdown():
    print ('ros shutdown')

def main():
    rospy.init_node('Success_rate', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    
    df = pd.DataFrame(columns=['Model_name', 'success_rate','var_s','var_f'] )

    #########  Load model       #######
    # BC1_filename = 'IMGP_model/BC/learner3.pickle'
    # BC2_filename = 'IMGP_model/BC/learner6.pickle'
    # BC3_filename = 'IMGP_model/BC/learner9.pickle'
    # BC4_filename = 'IMGP_model/BC/learner12.pickle'
    # BC5_filename = 'IMGP_model/BC/learner15.pickle'
    # BC6_filename = 'IMGP_model/BC/learner18.pickle'
    # BC7_filename = 'IMGP_model/BC/learner21.pickle'
    # BC8_filename = 'IMGP_model/BC/learner24.pickle'
    # BC9_filename = 'IMGP_model/BC/learner27.pickle'
    # BC10_filename = 'IMGP_model/BC/learner30.pickle'
    # DART1_filename = 'IMGP_model/DART/learner3.pickle'
    # DART2_filename = 'IMGP_model/DART/learner6.pickle'
    # DART3_filename = 'IMGP_model/DART/learner9.pickle'
    # DART4_filename = 'IMGP_model/DART/learner12.pickle'
    # DART5_filename = 'IMGP_model/DART/learner15.pickle'
    # DART6_filename = 'IMGP_model/DART/learner18.pickle'
    # DART7_filename = 'IMGP_model/DART/learner21.pickle'
    # DART8_filename = 'IMGP_model/DART/learner24.pickle'
    # DART9_filename = 'IMGP_model/DART/learner27.pickle'
    # DART10_filename = 'IMGP_model/DART/learner30.pickle'

    # BC1_filename = 'OMGP_model/BC/learner1.pickle'
    # BC2_filename = 'OMGP_model/BC/learner2.pickle'
    # BC3_filename = 'OMGP_model/BC/learner3.pickle'
    # BC4_filename = 'OMGP_model/BC/learner4.pickle'
    # BC5_filename = 'OMGP_model/BC/learner5.pickle'
    # DART1_filename = 'OMGP_model/DART/learner1.pickle'
    # DART2_filename = 'OMGP_model/DART/learner2.pickle'
    # DART3_filename = 'OMGP_model/DART/learner3.pickle'
    # DART4_filename = 'OMGP_model/DART/learner4.pickle'
    # DART5_filename = 'OMGP_model/DART/learner5.pickle'

    BC_filename = 'IMGP_model/BC/learner12.pickle'
    DART_filename = 'IMGP_model/DART/learner12.pickle'

    # models = [BC1_filename,BC2_filename,BC3_filename,BC4_filename,BC5_filename
    #           ,BC6_filename,BC7_filename,BC8_filename,BC9_filename,BC10_filename
    #           ,DART1_filename,DART2_filename,DART3_filename,DART4_filename,DART5_filename
    #           ,DART6_filename,DART7_filename,DART8_filename,DART9_filename,DART10_filename]
    # models = [DART1_filename,DART2_filename,DART3_filename,DART4_filename,DART5_filename
    #           ,DART6_filename,DART7_filename,DART8_filename,DART9_filename,DART10_filename]
    models = [DART_filename,BC_filename]
    #########  model selection  #######
    for filename in models:
        model = joblib.load(filename)
        #max trial definition
        '''we want to 100 trial per one mixture.'''
        Max_mixture = model.M
        Max_trial = Max_mixture * 500
        #mean of Variance
        Sum_ss = 0
        Sum_fs = 0
        N_s = 0
        N_f = 0
        var_s=0
        var_f=0
        successRate = 0
        trial = 0
        num_Mixture = 0
        #########  100s trial       #######
        while (trial < Max_trial) :
            trial += 1
            step=0
            temp_ss = 0
            finish_flag = False
            sampling_flag = False
            Sub.success = False
            fail_flag = False
            if trial > int((num_Mixture+1)*(Max_trial)/Max_mixture): num_Mixture += 1
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
                # ss, k = ss_action.max(0)
                # k = int(k)
                
                a_x = mm_action[num_Mixture][0][0]
                a_y = mm_action[num_Mixture][0][1]
                ss = ss_action[num_Mixture]
                # Constraint for safeF
                if abs(a_y) > 0.9 :
                    a_y /= abs(a_y)
                    a_y *= 0.9
                IMGP_a = [a_y,a_x]   
                a = IMGP_a
                
            #### Control
                #### Start
                if start_flag :
                    time = int(random.uniform(-1,12))
                    Pub.reset(time)
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
            print('Model : %s , Trial : %i, s : %i, f : %i'%(filename,trial, N_s,N_f))
        successRate = N_s/Max_trial
        if N_s > 0:
            var_s = Sum_ss/N_s
        else : var_s = 0
        if N_f > 0:
            var_f = Sum_fs/N_f

        else : var_f = 0
        
        df2 = pd.DataFrame({"Model_name": filename, 
                        "Num_Mixture" : Max_mixture,
                        "success_rate":successRate,
                        "var_s":var_s,
                        "var_f":var_f}) 
        df = df.append(df2, ignore_index=True)
        
        print(df)
        df.to_excel('data/success_rate/result.xlsx', sheet_name = 'new_sheet_name')
    


if __name__=='__main__':
    main()
