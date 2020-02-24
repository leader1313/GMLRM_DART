from geometry_msgs.msg import Point


class Fail(object):
    def __init__(self):
        self.fail_flag = False
        self.timeLimit = 20.0
        self.X_Limit = [-0.1,-1.2] #[max , min]
        self.Y_Limit = [-0.5, 0.3]

    def inside_Range(self,target_pose):
        fail_flag = False
        if (target_pose.x > self.X_Limit[0]) or (target_pose.x < self.X_Limit[1]) or (target_pose.y < self.Y_Limit[0]) or (target_pose.y > self.Y_Limit[1]) :
            fail_flag = True
        return fail_flag

    def time_condition(self,simulationTime):
        fail_flag = False
        if simulationTime > self.timeLimit :
            fail_flag = True
        else : fail_flag = False

        return fail_flag

    def fail_check(self, target_pose, simulationTime):
        if self.inside_Range(target_pose) or self.time_condition(simulationTime) :
            self.fail_flag = True
        else: self.fail_flag = False
        return self.fail_flag

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
    fail = Fail()
    while True:
        print(Sub.target_pose)
        Pub.input()
        fail_flag = fail.fail_check(Sub.target_pose,Sub.simulationTime)
        print(fail_flag)
        rate.sleep()
    rospy.spin()


if __name__=='__main__':
    main()
