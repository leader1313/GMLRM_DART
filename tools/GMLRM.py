#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det
from scipy.linalg import pinv

class GMLRM:
    def __init__(self, X, Y, K, M, T):
        #Constant
        self.X = X
        self.Y = Y
        self.D = X.shape[1]
        self.N = X.shape[0]
        self.K = K
        self.M = M
        self.T = T
        self.Phi = np.zeros((self.N,self.M))
        self.phi_mean = np.zeros(self.D)+1
        self.phi_sigma = np.diag((np.zeros(self.D)+1))
        self.Init_phi()
        
        #Variable
        self.prior = np.random.rand(self.K)                   # 어느 클래스에 속할 확률 vector
        self.Weight = np.random.rand(self.M,self.K)           # mean vector
        self.var = np.zeros(1)+1                              # variance
        self.r = np.zeros((self.N,K))                              # responsibility
        self.R = np.zeros((K,self.N,self.N))

    #=============== Support method =================
    def dot(self,x,y,z):
        a = np.dot(x,y)
        b = np.dot(a,z)
        return b

    def uni_Gaussian_Distribution(self,x, mean, var):
        E = np.exp(-1/(2*(var))*((x-mean)**2))
        y = (1/(((2*np.pi)*(var)**(1/2)))) * E
        return y

    def Gaussian_Distribution(self,x, D, mean, sigma):
        E = np.exp((-1/2)*self.dot((x-mean).T,(inv(sigma)),(x-mean)))
        y = (1/((2*np.pi)**(D/2)))*(1/((det(sigma))**(1/2)))*E        
        return y

    def uni_Gaussian_bias(self,x, mean, sigma):
        B = np.exp((-1/(2*((sigma)**2)))*((x-mean)**2))
        return B

    def Gaussian_bias(self,x, mean, sigma):
        B = np.exp((-1/2)*self.dot((x-mean).T,(inv(sigma)),(x-mean)))
        return B
    #phi : bias
    def cal_phi(self, X, m):
        phi = self.Gaussian_bias(X,m*self.phi_mean,self.phi_sigma)
        return phi

    #=============== Update method =================
    def Init_phi(self):
        for m in range(self.M):
            for n in range(self.N):
                self.Phi[n,m] = self.cal_phi(self.X[n], m)

    def responsibility(self, n, k):
        t = self.Y[n]
        p = self.prior
        wk_bn = np.dot(self.Weight[:,k].T,self.Phi[n])
        
        sum_r = np.zeros(1)[...,None]
        r_k = p[k]*self.uni_Gaussian_Distribution(t, wk_bn, self.var)[...,None]
        for j in range(self.K):
            wj_bn = np.dot(self.Weight[:,j].T, self.Phi[n])
            r1 = p[j]*self.uni_Gaussian_Distribution(t, wj_bn, self.var)
            sum_r += r1

        return r_k/sum_r

    def expectation(self): 
        for n in range(self.N):
            for k in range(self.K):
                self.r[n,k] = self.responsibility(n, k)
                self.R[k,n,n] = self.r[n,k]
        print(self.r)

    def maximization(self):
        sum_r = np.zeros(self.K)
        sum_rd = np.zeros(1)
        for k in range(self.K):
            for n in range(self.N):
                re = self.r[n,k]
                sum_r[k] += re
            self.prior[k] = sum_r[k]/self.N            
            invPRP = pinv(self.dot(self.Phi.T,self.R[k],self.Phi))
            PtRy = self.dot(self.Phi.T,self.R[k],self.Y)
            
            self.Weight[:,k][:,None] = np.dot(invPRP, PtRy)
            for n in range(self.N):
                re = self.r[n,k]
                d = (self.Y[n]-np.dot(self.Weight[:,k].T, self.Phi[n]))**2

                sum_rd += re * d
        self.var = sum_rd/self.N

    def EM(self):
        for t in range(self.T):
            self.expectation()
            self.maximization()

    def predict(self, new_X):
        new_phi = np.zeros(self.M)
        predict = np.zeros(self.K)
        X = new_X
        for m in range(self.M):
            new_phi[m] = self.cal_phi(X,m)
        for k in range(self.K):
            predict[k] = np.dot(self.Weight[:,k].T,new_phi)
        return predict
    

# #=========================sample data 정의================================

# N = 100
# n = int(N/2)
# X1 = np.linspace(0, np.pi*2, n)[:,None]
# Y1 = np.sin(X1) + np.random.randn(n)[:,None] * 0.1
# X2 = np.linspace(0, np.pi*2, n)[:,None]
# Y2 = np.cos(X2) + np.random.randn(n)[:,None] * 0.1

# X = np.append(X1,X2, axis = 0)
# Y = np.append(Y1,Y2, axis = 0)

# # True
# true_y1 = np.sin(X1)
# true_y2 = np.cos(X2)




# #==================== random initialize parameter =======================
# T=10
# K = 2                                       # model 수
# M = 7                                       # Number of model
# GM = GMLRM(X,Y,K,M,T)
# print("="*40)
# test_num = 100
# predict1 = np.zeros(test_num)[:,None]
# predict2 = np.zeros(test_num)[:,None]
# test_x = np.linspace(0, np.pi *2, test_num)[:, None]
# Pi = np.zeros((test_num,M))
# for m in range(M):
#     for n in range(test_num):
#         Pi[n,m] = GM.Gaussian_bias(test_x[n],m*GM.phi_mean,GM.phi_sigma)
# Weight ,var = GM.EM() 
# for n in range(test_num):
#     predict = np.zeros(K)
#     predict = GM.predict(test_x[n],Weight)
#     predict1[n] = predict[0]
#     predict2[n] = predict[1]
# ss = np.sqrt(var)


# plt.figure()
# plt.plot(X, Y, "g*")
# plt.plot(X1, true_y1, 'k')
# plt.plot(X2, true_y2, 'k')
# line1 = plt.plot(test_x, predict1, 'b')
# plt.plot(test_x, predict1 + ss,"--", color = line1[0].get_color() )
# plt.plot(test_x, predict1 - ss,"--", color = line1[0].get_color() )

# line2 = plt.plot(test_x, predict2, 'r')
# plt.plot(test_x, predict2 + ss,"--", color = line2[0].get_color() )
# plt.plot(test_x, predict2 - ss,"--", color = line2[0].get_color() )

# plt.show()


    
