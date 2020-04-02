#-*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det, pinv
import sys,random


class GMLRM:
    def __init__(self, X, Y, K, M, T):
        #Constant
        self.X = X
        self.Y = Y
        self.D = X.shape[1]
        self.O = Y.shape[1]
        self.N = X.shape[0]
        self.Noise = np.zeros(self.O)
        self.K = K
        self.M = M
        self.Num_Model = (self.M)**(self.D)
        self.T = T
        self.Phi = np.zeros((self.N,self.Num_Model))
        #============example============
        # self.X_Max = np.array([2*np.pi])
        # self.X_Min = np.array([-np.pi])
        # self.sigma = np.array([1])
        #==========6D=============
        # self.X_Max = np.array([-0.35, -0.35, 0.24, 0.24, -0.35, 0.24])
        # self.X_Min = np.array([-1.1, -1.1, -0.42, -0.42, -1.1, -0.42])
        # self.sigma = np.array([1,1,1,1,1,1])
        #==========4D=============
        # self.X_Max = np.array([-0.4, 0.2, -0.4, 0.2])
        # self.X_Min = np.array([-1.1, -0.5, -1.1, -0.5])
        # self.sigma = np.array([1,1,1,1])
        #==========3D=============
        self.X_Max = np.array([0.2, 0.2, 0.2])
        self.X_Min = np.array([-0.5, -0.5, -0.5])
        self.sigma = np.array([1,1,1])
        self.phi_sigma = np.diag(self.sigma)

        self.range = np.zeros(self.D)
        self.biasRange()
        self.Init_phi()
        
        #Variable
        self.prior = np.zeros(self.K)+(1/self.K)                   # 어느 클래스에 속할 확률 vector
        self.Wvar = np.sqrt(1/(self.K*self.Num_Model*self.O))
        # self.Weight = self.Wvar*np.random.randn(self.K,self.Num_Model,self.O)   # mean vector
        self.Weight = np.random.uniform(-self.Wvar,self.Wvar,size=(self.K,self.Num_Model,self.O))
        self.var = np.diag(np.zeros(self.O)+1)                              # variance
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

    def Gaussian_Distribution(self,x, D, mean, cov):
        x_m = (x-mean)
        E = np.exp((-1/2)*self.dot((x_m).T,(inv(cov)),(x_m)))
        y = 1. / (np.sqrt((2 * np.pi)**D * det(cov))) *E                    
        return y

    def uni_Gaussian_bias(self,x, mean, sigma):
        B = np.exp((-1/(2*((sigma)**2)))*((x-mean)**2))
        return B

    def Gaussian_bias(self,x, mean, cov):
        B = np.exp((-1/2)*self.dot((x-mean).T,(inv(cov)),(x-mean)))
        return B
    #phi : bias
    def cal_phi(self, X, m):
        x = (np.zeros(self.D)+1)
        x = self.Base_10_to_n(x, m, 0)
        mean = self.X_Min + np.dot(self.range,np.diag(x))
        sigma = self.phi_sigma
        phi = self.Gaussian_bias(X, mean, sigma)
        return phi

    def Base_10_to_n(self, X ,n, i):
        if (int((n)/(self.M))):
            X[i] = (n)%(self.M)+1
            self.Base_10_to_n(X,int((n)/(self.M)), i+1)
        else : X[i] = (n)%(self.M)+1
        return X

    #=============== Update method =================
    def biasRange(self):
        self.range = (self.X_Max-self.X_Min)/(self.M + 1)
        
    def Init_phi(self):
        for n in range(self.N):
            for m in range(self.Num_Model):
                self.Phi[n,m]=self.cal_phi(self.X[n],m)
            #Normalization for basis function
            Phi_norm = sum(self.Phi[n,:])
            self.Phi[n,:] /= Phi_norm
            

    def responsibility(self, n, k):
        t = self.Y[n][...,None]
        p = self.prior
        wk_bn = np.dot(self.Weight[k].T,self.Phi[n])[...,None]
        sum_r = np.zeros(1)[...,None]
        r_k = p[k]*self.Gaussian_Distribution(t ,self.O , wk_bn, self.var)
        for j in range(self.K):
            wj_bn = np.dot(self.Weight[j].T, self.Phi[n])[...,None]
            r1 = p[j]*self.Gaussian_Distribution(t,self.O, wj_bn, self.var)
            sum_r += r1
            
        return r_k/sum_r

    def expectation(self): 
        for n in range(self.N):
            for k in range(self.K):
                self.r[n,k] = self.responsibility(n, k)
                self.R[k,n,n] = self.r[n,k]
        

    def maximization(self):
        sum_r = np.zeros(self.K)
        sum_rd = np.diag(np.zeros(self.O))
        for k in range(self.K):
            for n in range(self.N):
                re = self.r[n,k]
                sum_r[k] += re
            self.prior[k] = sum_r[k]/self.N        
            invPRP = pinv(self.dot(self.Phi.T,self.R[k],self.Phi))
            PtRy = self.dot(self.Phi.T,self.R[k],self.Y)
            
            self.Weight[k] = np.dot(invPRP, PtRy)
            for n in range(self.N):
                re = self.r[n,k]
                d = (self.Y[n]-np.dot(self.Weight[k].T, self.Phi[n]))[None,...]
                D = np.dot(d.T,d)
                D = re * D
                sum_rd += D
                
        var = np.diag(sum_rd/self.N)
        self.var = np.diag(var)
        
    def learning(self):
        for t in range(self.T):
            print(self.prior)
            self.expectation()
            self.maximization()
        self.Noise = self.var
                   

    def predict(self, new_X):
        new_phi = np.zeros(self.Num_Model)
        predict = np.zeros((self.K,self.O))
        X = new_X
        for m in range(self.Num_Model):
            new_phi[m] = self.cal_phi(X,m)
        Phi_norm = sum(new_phi)
        new_phi /= Phi_norm
        for k in range(self.K):
            predict[k] = np.dot(self.Weight[k].T,new_phi)
        return predict
    

# #=========================sample data 정의================================

# N = 100
# n = int(N/4)
# X3 = np.linspace(-np.pi,0, n)[:,None]
# X4 = np.linspace(np.pi,2*np.pi, n)[:,None]
# Y3 =  np.random.randn(n)[:,None] * 0.1
# Y4 =  np.random.randn(n)[:,None] * 0.1
# X1 = np.linspace(0, np.pi, n)[:,None]
# Y1 = np.sin(X1) + np.random.randn(n)[:,None] * 0.1
# X2 = np.linspace(0, np.pi, n)[:,None]
# Y2 = -np.sin(X2) + np.random.randn(n)[:,None] * 0.1

# x1 = np.append(X1,X2, axis = 0)
# x2 = np.append(X3,X4, axis = 0)
# y1 = np.append(Y1,Y2, axis = 0)
# y2 = np.append(Y3,Y4, axis = 0)
# X = np.append(x1,x2, axis = 0)
# X = np.append(x1,x2, axis = 0)
# Y = np.append(y1,y2, axis = 0)

# # True
# true_y1 = np.sin(X1)
# true_y2 = -np.sin(X2)





# #==================== random initialize parameter =======================
# T=10
# K = 2                                       # model 수
# M = 7                                       # Number of model
# GM = GMLRM(X,Y,K,M,T)
# print("="*40)
# test_num = 100
# predict1 = np.zeros(test_num)[:,None]
# predict2 = np.zeros(test_num)[:,None]
# test_x = np.linspace(-np.pi, 2*np.pi, test_num)[:, None]

# GM.learning()
# for n in range(test_num):
#     predict = np.zeros(K)
#     predict = GM.predict(test_x[n])
#     predict1[n] = predict[0]
#     predict2[n] = predict[1]
# ss = np.sqrt(GM.var)


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


    
