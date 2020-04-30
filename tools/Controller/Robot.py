import numpy as np


class Robot(object):
    def __init__(self):
        [self.E_x,self.IE_x] = np.zeros(2)
        [self.E_y,self.IE_y] = np.zeros(2)

    def PIDcontrol(self,goal,end,DE,IE):
        P = 5
        I = 1
        D = 5
        E = goal-end
        DE = E-DE
        
        if abs(E) > 0.05 :
            a = (E/abs(E))*1.0
        else :
            IE += E
            a = P*E-D*DE+I*IE

        return a, E, IE

    def policy(self,s,k):
        a_x,self.E_x,self.IE_x = self.PIDcontrol(s[k].x,s[2].x,self.E_x,self.IE_x)
        a_y,self.E_y,self.IE_y = self.PIDcontrol(s[k].y,s[2].y,self.E_y,self.IE_y)
        return a_x,a_y