import numpy as np
from numpy.linalg import inv, det
from scipy.optimize import minimize
import math,torch,sys

class Noise:
    def __init__(self, state, sup_intended_action, ler_policy, noise):
        self.s = state
        self.N = self.s.shape[0]
        self.a_sup = sup_intended_action
        self.ler = ler_policy
        self.O = self.a_sup.shape[1]
        self.K = self.ler.K
        self.k = np.zeros(self.N)
        self.a_ler = np.zeros((self.K,self.N,self.O))
        self.nll = np.zeros(1)
        self.noise = noise
        self.noise = torch.tensor(self.noise)
        print(self.noise)
        self.robot_action()
        self.classification()
        self.negative_log_likelihood(self.noise)
        self.b = (0.000001,0.1)
        self.bnds = (self.b,self.b)
    
    def dot(self,x,y,z):
        a = np.dot(x,y)
        b = np.dot(a,z)
        # a = torch.mm(x,y)
        # b = torch.mm(a,z)
        return b

    def compute_grad(self, flag):
        self.noise.requires_grad = flag
    
    def robot_action(self):
        for k in range(self.K):
            for n in range(self.N):
                for o in range(self.O):
                    self.a_ler[k,n,o] = self.ler.predict(self.s[n])[k][o]

    def classification(self):
        for n in range(self.N):
            d1 = (self.a_ler[0,n]-self.a_sup[n])[...,None]
            d1 = np.dot(d1.T,d1)
            d2 = (self.a_ler[1,n]-self.a_sup[n])[...,None]
            d2 = np.dot(d2.T,d2)
            if (d1-d2) > 0 :
                self.k[n] = 1
            else: self.k[n] = 0

    # def Negativell(self):
    #     Nll = 0
    #     E = 0
    #     cov = torch.diag(self.noise)
    #     Det = torch.det(cov)
    #     for n in range(self.N):
    #         k = int(self.k[n])
    #         x_m = torch.from_numpy(self.a_ler[k,n]-self.a_sup[n])[None,...]
    #         e = (1/2)*self.dot((x_m),(torch.inverse(cov)),(x_m).T)
    #         E += e
    #     Nll = (self.N/2)*(self.O*math.log(2*math.pi)+math.log(Det)) + E
    #     self.nll = Nll
    #     return Nll
        
    def negative_log_likelihood(self, noise):
        Nll = 0
        E = 0
        cov = np.diag(noise)
        for n in range(self.N):
            k = int(self.k[n])
            x_m = (self.a_ler[k,n]-self.a_sup[n])
            e = (1/2)*self.dot((x_m).T,(inv(cov)),(x_m))
            E += e
            
        Nll = (self.N/2)*(self.O*math.log(2*math.pi)+math.log(det(cov))) + E
        self.nll = Nll
        return Nll

    def optimize(self):
        # max_iter = 100
        # post_nll = self.Negativell()
        # optimized_noise = self.noise
        # self.compute_grad(True)
        # param = [self.noise]
        # learning_rate = 1e-5
        # optimizer = torch.optim.Adam(param, lr=learning_rate)
        print('pre Noise = ',self.noise, end =" ")
        print('NLL = %f'%(self.nll))
        sol = minimize(self.negative_log_likelihood, self.noise,
                        method='SLSQP', bounds=self.bnds)
        self.noise = sol.x
        # for i in range(max_iter):
        #     optimizer.zero_grad()
        #     f = self.Negativell()
        #     f.backward()
        #     optimizer.step()
        print('Update Noise = ',self.noise, end =" ")
        print('NLL = %f'%(self.nll))
        # self.compute_grad(False)
            
        
 
    




