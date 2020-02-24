import time
import rospy, torch
from geometry_msgs.msg import Point
from std_msgs.msg import Float32, Bool, Float32MultiArray



class Subscriber(object):
    def __init__(self):
        self.goal_1 = Point()
        self.goal_2 = Point()
        self.endeffector_pose = Point()
        self.endeffector_vel = Point()
        self.target_pose = Point()
        self.post_pose = Point()
        self.simulationTime = Float32().data
        self.success = Bool()
        
        #SUBSCRIBER
        rospy.Subscriber('/goal_1/point',        Point,             self._callback_goal1)
        rospy.Subscriber('/goal_2/point',        Point,             self._callback_goal2)
        rospy.Subscriber('/endeffector/Point', Point,             self._callback_endeffector_pose)
        rospy.Subscriber('/Target/Point',      Point,             self._callback_target_pose)
        rospy.Subscriber('/simulationTime',    Float32,           self._callback_simulationTime)
        rospy.Subscriber('/Success',           Bool,              self._callback_success)

    def _callback_goal1(self, data):
        self.goal_1 = data

    def _callback_goal2(self, data):
        self.goal_2 = data

    def _callback_endeffector_pose(self, data):
        self.endeffector_pose = data

    def _callback_target_pose(self, data):
        self.target_pose = data

    def _callback_simulationTime(self, data):
        self.simulationTime = data.data

    def _callback_success(self, data):
        self.success = data.data


def shutdown():
    print ('ros shutdown')
    
def main():
    rospy.init_node('Data', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    DT = Subscriber()
    while True:
        print(DT.target_pose)
        rate.sleep()
    rospy.spin()


if __name__=='__main__':
    main()
