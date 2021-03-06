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
    BC_filename = []
    DART_filename = []
    for i in range(10):
        filename = 'IMGP_model/BC/learner'+str(i+1)+'.pickle'
        BC_filename += [filename]

    for j in range(10):
        filename = 'HIMGP_model/DART/learner'+str(j+1)+'.pickle'
        DART_filename += [filename]
    
    models = BC_filename + DART_filename
    
    print(models)
    
    models = ['HIMGP_model/DART/learner9.pickle']

    #########  model selection  #######
    for filename in models:
        model = joblib.load(filename)
        #max trial definition
        '''we want to 100 trial per one mixture.'''
        Max_mixture = model.M

        Max_trial = Max_mixture * 100
        #mean of Variance
        Sum_ss = 0
        Sum_fs = 0
        N_s = 0.0
        N_f = 0.0
        var_s=0.0
        var_f=0.0
        successRate = 0.0
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

                s = [s[0].x,s[0].y,s[1].x,s[1].y,s[2].x,s[2].y]

                temp_state = s
                np_temp = np.asarray(temp_state)[None,...]
                te_temp = torch.from_numpy(np_temp).float()
                mm_action, ss_action = model.predict(te_temp)

            #### Action and regret
                # ss, k = ss_action.max(0)
                # k = int(k)
                noise = torch.tensor(np.random.normal(0,0.5,1))
                a_x = mm_action[num_Mixture][0][0]
                a_y = mm_action[num_Mixture][0][1]
                ss = ss_action[num_Mixture]
         
                action = [a_y,a_x]   
                
            #### Control
                #### Start
                if start_flag :
                    time = int(random.uniform(-1,12))
                    Pub.reset(time)
                    rate.sleep()
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

                if (Sub.simulationTime > 0.1) and (sampling_flag) :
                    step += 1
                    temp_ss += ss
                    Pub.actionInput(action)

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
