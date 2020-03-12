import numpy as np
import torch, math


class Supervisor():
    def __init__(self, cov):
        self.cov = cov
        self.sigma = math.sqrt(cov)

    def sample_action(self,action):
        in_a = action
        sampled_action = np.random.normal(0,self.sigma,1) + in_a
        return sampled_action

    def intended_action(self, action):
        return action


def shutdown():
    print ('ros shutdown')
    
def main():
    from Data_Subscriber import Subscriber
    from Data_Publisher import Publisher
    
    import rospy
    rospy.init_node('Data', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    Sub = Subscriber()
    Pub = Publisher()
    Sup = Supervisor(0.02)
    while True:
        axes, buttons = Pub.joyInput()
        Pub.actionInput(Sup.sample_action(axes))
        
        
        rate.sleep()
    rospy.spin()


if __name__=='__main__':
    main()
