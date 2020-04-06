import sys
sys.path.append('/home/hanbit-o/python/GMLRM_DART/tools/')
from GamePad import *
import os, time, rospy,random, math
import numpy as np
from geometry_msgs.msg import Point
from std_msgs.msg import Float32MultiArray, Bool


class Publisher(object):
    def __init__(self):
        self.game_pad = GamePad(0)
        self.game_pad.button_mode_[0] = 1
        self.game_pad.button_mode_[1] = 1
        self.game_pad.button_mode_[2] = 1
        self.game_pad.button_mode_[3] = 1

        self.game_pad_state = Float32MultiArray()
        self.action_pub = rospy.Publisher('Action', Float32MultiArray, queue_size = 10)
        self.reset_target_pub = rospy.Publisher('reset_target_pose', Point, queue_size = 10)
        self.reset_goal1_pub = rospy.Publisher('reset_goal1_pose', Point, queue_size = 10)
        self.reset_goal2_pub = rospy.Publisher('reset_goal2_pose', Point, queue_size = 10)
        self.stop_pub = rospy.Publisher('stopSimulation', Bool, queue_size = 1)
        self.start_pub = rospy.Publisher('startSimulation', Bool, queue_size = 1)
        self.pause_pub = rospy.Publisher('pauseSimulation', Bool, queue_size = 1)
        self.init_target = Point(x = -0.4155, y = -0.1098 , z = 0.34)
        self.goal_pose = Point()

    def reset(self):
        self.sim_stop()
        rospy.sleep(0.2)
        self.sim_start()
        rospy.sleep(0.5)
        self.reset_goal1_pub.publish(self.random_goal_pose(-0.75,-0.25))
        self.reset_goal2_pub.publish(self.random_goal_pose(-0.75,0.05))
        print('reset')
        rospy.sleep(0.5)

    def sim_start(self):
        self.start_pub.publish(True)
        print('start simulation...')

    def sim_stop(self):
        self.stop_pub.publish(True)
        print('stop simulation...')

    def sim_pause(self):
        self.pause_pub.publish(True)
        print('pause simulation...')

    def random_goal_pose(self,x,y):
        max_X_length = 0.3
        max_Y_length = 0.2
        # center_x = -0.7074
        # center_y = -0.0998
        center_x = x
        center_y = y
        x = center_x + random.uniform(-max_X_length/2,max_X_length/2)
        y = center_y + random.uniform(-max_Y_length/2,max_Y_length/2)
        
        self.goal_pose.x = x
        self.goal_pose.y = y
        self.goal_pose.z = self.init_target.z
        return self.goal_pose
        
        
    # def keyInput(self):
    #     if keyboard.is_pressed('right arrow'):
    #         self.axes[0] = 0.1
    #     elif keyboard.is_pressed('left arrow'):
    #         self.axes[0] = -0.1
    #     if keyboard.is_pressed('up arrow'):
    #         self.axes[1] = 0.1
    #     elif keyboard.is_pressed('down arrow'):
    #         self.axes[1] = -0.1
        
    #     if keyboard.is_pressed('s'):
    #         self.buttons[1] = True
    #     if keyboard.is_pressed('a'):
    #         self.buttons[2] = True
    #     return self.axes, self.buttons

    def joyInput(self):
        self.game_pad.Update()
        self.axes = self.game_pad.axes_[:2]
        self.buttons = self.game_pad.buttons_[:4]
        return self.axes, self.buttons

    def actionInput(self, action):
        # action += np.random.normal(0,0.1,1)
        self.game_pad_state.data = list(action)
        self.action_pub.publish(self.game_pad_state)


def shutdown():
    print ('ros shutdown')
    
def main():
    rospy.init_node('Data', anonymous=True, disable_signals=True)
    rospy.on_shutdown(shutdown)
    rate = rospy.Rate(10)
    DT = Publisher()
    
    while True:
        print('axes')
        axes, buttons = DT.joyInput()
        print(axes)
        print('button')
        print(buttons)
        DT.actionInput(axes)

        rate.sleep()
    rospy.spin()


if __name__=='__main__':
    main()
