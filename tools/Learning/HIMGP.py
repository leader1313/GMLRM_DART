import numpy as np
import torch
import copy

class HIMGP:
    def __init__(self, X, Y, M, iters, kernel, old_sigma = [], K = []):
        '''
        Argument
                    X : State
                    Y : Action
                    N : Number of data
                    D : Dimension of Action
                    T : Max iteration
                    K : Current Noise update iteration
            prob_thresh : Threshhold of allocate probability z
            Psi : Heteroscedastic gaussian noise variance (N X 1)vector
            old_Psi : Previous Heteroscedastic gaussian noise variance (N' X 1)vector
            new_Psi : New Heteroscedastic gaussian noise variance ((N-N') X 1)vector 
                      that we want to update
        '''
        self.X = X
        self.Y = Y
        self.N = self.X.shape[0]
        self.M = M
        self.D = self.Y.shape[1]
        self.T = iters
        self.prob_thresh = 0.001
        self.Noise = np.zeros(self.D)
        self.kern = np.array([kernel() for _ in range(self.M)])
        self.alpha = torch.tensor(np.log(100)).float()
        self.K = len(K)+1
        self.N_K = K
        # if old_sigma == [] :
        self.log_sigma = [torch.tensor(np.log(0.4)).float() for _ in range(self.K)]
        # else:
        #     self.log_sigma =  old_sigma
        #     self.log_sigma += [torch.tensor(np.log(0.1)).float()]
        self.Psi = self.init_Psi()

        self.q_z_pi = torch.ones(self.M, self.N) / self.M

        self.v_beta_a = torch.ones(self.M)
        self.v_beta_b = torch.ones(self.M)*torch.exp(self.alpha)

        self.q_f_mean = torch.tensor(np.random.normal(0, 0.09, (self.M, self.N, self.D))).float()
        self.q_f_sig = torch.stack([torch.eye(self.N) for m in range(self.M)])

    def init_Psi(self):
        if self.N_K == [] :
            Psi = [torch.ones(self.N,1)*self.log_sigma[0].exp()]
            self.N_K += [self.N]
        else:
            old_N = 0
            for i in range(self.K-1): 
                old_N += self.N_K[i]
            new_N = self.N-old_N
            self.N_K += [new_N]
            Psi = torch.cat([(torch.ones(self.N_K[k],1)*self.log_sigma[k].exp())\
                 for k in range(self.K)])
        return Psi
    
    def new_Psi(self):
        Psi = torch.cat([(torch.ones(self.N_K[k],1)*self.log_sigma[k].exp())\
                 for k in range(self.K)])
        return Psi

    def update_q_z(self):
        '''
            Psi : Noise matrix (N X 1)
            E_ln_v
            E_ln_1_minus_v
            ln_rho : (M X N) matrix
        '''
        Psi = self.new_Psi()
        
        E_ln_v = torch.digamma(self.v_beta_a) - torch.digamma(self.v_beta_a+self.v_beta_b)
        E_ln_1_minus_v = torch.digamma(self.v_beta_b) - torch.digamma(self.v_beta_a+self.v_beta_b)

        tmp_sum = torch.zeros(self.M)
        for m in range(0, self.M):
            tmp_sum[m] += E_ln_v[m]
            for i in np.arange(0, m):
                tmp_sum[m] += E_ln_1_minus_v[i]
        
        ln_rho = -0.5 * ((((self.Y.repeat(self.M,1,1)-self.q_f_mean)**2)/Psi).sum(2) \
                    + torch.stack([torch.diag(self.q_f_sig[m]/Psi) for m in range(self.M)]) \
                    + self.D*torch.log(np.pi*2*Psi).repeat(self.M,1,1).sum(2)) \
                    + (tmp_sum).repeat(self.N,1).T

        self.q_z_pi = torch.exp(ln_rho)
        self.q_z_pi /= self.q_z_pi.sum(0)[None,:]
        self.q_z_pi[torch.isnan(self.q_z_pi)] = 1.0/self.M
        # n_digits = 3
        # self.q_z_pi = (self.q_z_pi * 10**n_digits).round() / (10**n_digits)

    def update_q_f(self):
        '''
        Psi : Noise matrix (N X 1)
        E_ln_v
        E_ln_1_minus_v
        ln_rho : (M X N) matrix
        '''
        Psi = self.new_Psi()
        K = torch.stack([self.kern[m].K(self.X)+torch.eye(self.N)*1e-5 for m in range(self.M)])
    
        sqrtK = torch.sqrt(K)
        B = torch.stack([torch.diag(self.q_z_pi[m]/Psi.sum(1)) for m in range(self.M)])

        self.q_f_sig = sqrtK.bmm(torch.solve(sqrtK, torch.stack([torch.eye(self.N) for _ in range(self.M)])+sqrtK.bmm(B).bmm(sqrtK))[0])
        self.q_f_mean = self.q_f_sig.bmm(B).bmm(self.Y.repeat(self.M,1,1))

    def update_q_v(self):
        alpha = torch.exp(self.alpha)
        for m in range(0,self.M):
            self.v_beta_a[m] = 1.0 + self.q_z_pi[m,:].sum()
            tmpSum = torch.zeros(1)
            for j in range(m+1,self.M):
                tmpSum += self.q_z_pi[j,:].sum()

            self.v_beta_b[m] = alpha + tmpSum

    def predict(self, x):
        Psi = self.new_Psi()
        
        K = torch.stack([self.kern[m].K(self.X)+torch.eye(self.N)*1e-5 for m in range(self.M)])

        B = torch.stack([torch.diag((self.q_z_pi[m][...,None]/Psi).sum(1)) for m in range(self.M)])
        sqrtB = torch.sqrt(B)
        R = torch.stack([torch.eye(self.N)+sqrtB[m].mm(K[m]).mm(sqrtB[m]) for m in range(self.M)])

        Kx = torch.stack([self.kern[m].K(x) for m in range(self.M)])
        Knx = torch.stack([self.kern[m].K(self.X, x) for m in range(self.M)])

        mean = Knx.permute(0,2,1).bmm(sqrtB).bmm(torch.solve(sqrtB, R)[0]).bmm(self.Y.repeat(self.M,1,1))
        sigma = torch.stack([torch.diag(Kx[m] - Knx[m].t().mm(sqrtB[m]).mm(torch.solve(sqrtB[m], R[m])[0]).mm(Knx[m]))+Psi[0] for m in range(self.M)])
        sigma[sigma<0.00001]=0.00001
        return mean, sigma

    def compute_grad(self, flag):
        self.alpha.requires_grad = flag
        for k in range(self.K):
            self.log_sigma[k].requires_grad = flag
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                self.kern[m].compute_grad(flag)

    def expectation(self, N=10):
        for i in range(N):
            self.update_q_f()
            self.update_q_v()
            self.update_q_z()
        
    def save_checkpoint(self):
        log_sigma = []
        kern = []
        alpha = [self.alpha]
        log_sigma += self.log_sigma
        q_z_pi = self.q_z_pi
        for m in range(self.M):
            theta = self.kern[m].param()
            kern += theta
        torch.save({
                    'q_z_pi':q_z_pi
                    ,'alpha':alpha
                    ,'log_sigma':log_sigma
                    ,'kern':kern
                    }
                    ,'data/checkpoint.pt')

    def load_checkpoint(self):
        checkPoint = torch.load('data/checkpoint.pt')

        self.q_z_pi = checkPoint['q_z_pi']
        self.alpha = checkPoint['alpha']
        self.log_sigma = checkPoint['log_sigma']
        kern = checkPoint['kern']
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                self.kern[m].sigma = (kern[m])
               

    def upperBound(self, n_batch=None):
        
        if n_batch is None:
            ind = torch.arange(self.N)
        else:
            ind = torch.randperm(self.N)[:n_batch]
        
        alpha = torch.exp(self.alpha)
        Psi = self.new_Psi().sum(1).repeat(self.M,1)

#Expectation of likelihood        

        K = torch.stack([self.kern[m].K(self.X[ind])+torch.eye(ind.shape[0])*1e-5 for m in range(self.M)])
        q_z_pi = copy.deepcopy(self.q_z_pi)
        q_z_pi[q_z_pi!=0] /= Psi[q_z_pi!=0]
        
        B = torch.stack([torch.diag(q_z_pi[m][ind]) for m in range(self.M)])
        sqrtB = torch.sqrt(B)

        R = torch.stack([torch.eye(ind.shape[0])+sqrtB[m].mm(K[m]).mm(sqrtB[m]) for m in range(self.M)])

        lk_1 = torch.zeros(1)
        lk_2 = torch.zeros(1)
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                lk_1 += self.D*torch.slogdet(R[m])[1]
                lk_1 += 0.5*torch.trace(self.Y[ind].t().mm(sqrtB[m]).mm(torch.solve(sqrtB[m], R[m])[0]).mm(self.Y[ind]))

                lk_2 += 0.5*(self.q_z_pi[m]*torch.log(np.pi*2*Psi[m])).sum()
                

        E_zv = torch.zeros(1)
        kl_v = torch.zeros(1)

        v_beta_a = torch.ones(self.M)
        v_beta_b = torch.ones(self.M)*alpha

        for m in range(0,self.M):
            # if self.q_z_pi[m].sum() > self.prob_thresh:
            v_beta_a[m] = 1.0 + self.q_z_pi[m][ind].sum()
            tmpSum = torch.zeros(1)
            for i in range(m+1,self.M):
                tmpSum += self.q_z_pi[i][ind].sum()
            v_beta_b[m] = alpha + tmpSum
                
        E_ln_v = torch.digamma(v_beta_a) - torch.digamma(v_beta_a+v_beta_b)
        E_ln_1_minus_v = torch.digamma(v_beta_b) - torch.digamma(v_beta_a+v_beta_b)

# E_zv(p(z|v))======================================================================================
        tmp_sum = torch.zeros(self.M)    
        ln_rho = torch.zeros(ind.shape[0])
        r = copy.deepcopy(self.q_z_pi)
        r[r!=0] = torch.log(r[r!=0])

        for m in range(0,self.M):
            # if self.q_z_pi[m].sum() > self.prob_thresh:
            tmp_sum[m] += E_ln_v[m]
            for i in range(0, m-1):
                tmp_sum[m] += E_ln_1_minus_v[i]

            E_zv += (self.q_z_pi[m][ind]*(tmp_sum[m])).sum()
            E_zv -= (self.q_z_pi[m][ind][None,...].mm(r[m][ind][...,None])).sum()
                
# KL(v*|v)=============================================================================                
        for m in range(0,self.M):
            # if self.q_z_pi[m].sum() > self.prob_thresh:
            kl_v -= torch.lgamma(v_beta_a[m])+torch.lgamma(v_beta_b[m])-torch.lgamma(v_beta_a[m]+v_beta_b[m])
            kl_v += torch.lgamma(alpha)-torch.lgamma(1+alpha)
            kl_v += (v_beta_a[m]-1)*(E_ln_v[m])+(v_beta_b[m]-alpha)*(E_ln_1_minus_v[m])
                
        
        return (lk_1 + lk_2 - E_zv + kl_v)/self.N

    def learning(self):
        Max_step = self.T
        NL = self.upperBound()
        self.save_checkpoint()
        step = 0
        stop_flag = False
        Max_patient = 5
        patient_count = 0 
        while ((step < Max_step) and not(stop_flag)):
            step += 1
            print("=========================")
            print('E step')
            self.expectation(30)
            print('M step')
            self.maximization()
            
            print(step,' th NL : ',self.upperBound())
            print('Sigma : ',np.exp(self.log_sigma))
            print('Alpha : ',torch.exp(self.alpha))
            print("Z : ",self.q_z_pi.sum(axis = 1))
            if NL > self.upperBound():
                patient_count = 0
                NL = self.upperBound()
                self.save_checkpoint()
                
            else : 
                patient_count += 1
                print("-------Patient_Count(< %i) : %i"%(Max_patient,patient_count))
                if patient_count >= Max_patient: 
                    stop_flag = True
                    
        self.load_checkpoint()
        limit_prob = 0.5*self.N*(1/self.M)
        
        print(self.q_z_pi.sum(axis = 1))
        M = self.q_z_pi.sum(axis = 1) > limit_prob
        
        M = M.numpy()
        
        num_Mixture = M.sum()
            
        self.kern = self.kern[M]

        self.q_z_pi = self.q_z_pi[self.q_z_pi.sum(axis = 1) > limit_prob]
        self.M = num_Mixture
        self.Noise = np.exp(self.log_sigma)
        self.old_sigma = self.log_sigma
        print('-------------------------------------------')
        print('Number of Mixture : %i'%(num_Mixture))
        print('Optimized Noise : ',(self.Noise))
        

    def maximization(self):
        max_iter = 50
        
        self.compute_grad(True)
        param = [self.alpha]
        for k in range(self.K):
            param += [self.log_sigma[k]]

        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                param += self.kern[m].param()
        
        # optimizer = torch.optim.Adam(param,lr=0.01)
        optimizer = torch.optim.Adadelta(param,lr=1.0)
        # optimizer = torch.optim.SGD(param,lr=0.1,momentum=0.9)
        for i in range(max_iter):
            optimizer.zero_grad()

            try:
                f = self.upperBound(n_batch=10)
                f.backward()
            except (TypeError, RuntimeError) as e:
                print(e)
                import ipdb; ipdb.set_trace()

            optimizer.step()
            with torch.no_grad():
                for parameter in param:
                    min = torch.tensor(np.log(0.00005))
                    max = 1000
                    parameter.clamp_(min, max)
        self.compute_grad(False)
    
    
if __name__=="__main__":
    import sys
    from kernel import GaussianKernel
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    N = 50

    X1 = torch.linspace(0, torch.tensor(np.pi), N)[:,None]
    X2 = torch.linspace(torch.tensor(np.pi), torch.tensor(np.pi*2), N)[:,None]
    X3 = torch.linspace(torch.tensor(np.pi*2), torch.tensor(np.pi*3), N)[:,None]
    # X = torch.cat([X1,X2])
    X = torch.cat([X1,X2,X3]).float()
    # sigma = torch.tensor(0.2)
    
    Y1 = torch.sin(X1) + torch.randn(N)[:,None]*0.05
    Y2 = torch.sin(X2) + torch.randn(N)[:,None]*0.1
    Y3 = torch.sin(X3) + torch.randn(N)[:,None]*0.1
    Y1 = torch.cat([Y1, Y2,Y3]).float()
    Y2 = torch.cos(X1) + torch.randn(N)[:,None]*0.05
    Y3 = torch.cos(X2) + torch.randn(N)[:,None]*0.1
    Y4 = torch.cos(X3) + torch.randn(N)[:,None]*0.1
    Y2 = torch.cat([Y2, Y3,Y4]).float()
    # Y2 = torch.cos(X) + torch.randn(N)[:,None]*sigma
    # old_sigma = torch.tensor(0.01)
    # old_Psi = old_sigma.repeat(50,1)
    # old_Psi = None
    old_sigma = [torch.tensor(np.log(0.01)).float(),torch.tensor(np.log(0.1)).float()]
    # old_sigma = [torch.tensor(np.log(0.01)).float()]
    K = [100,100]
    # K = [100]
    kern = GaussianKernel()
    model = HIMGP(torch.cat([X,X]).float(), torch.cat([Y1, Y2]).float(), 5,5, GaussianKernel\
                    ,old_sigma=old_sigma,K = K)
    # model = HIMGP(torch.cat([X,X]).float(), torch.cat([Y1, Y2]).float(), 5,5, GaussianKernel)
    model.learning()
    # old_sigma = model.old_sigma
    # K = model.N_K
    # kern = GaussianKernel()
    # model = HIMGP(torch.cat([X,X]).float(), torch.cat([Y1, Y2]).float(), 5,5, GaussianKernel\
    #                 ,old_sigma=old_sigma,K = K)
    # model.learning()
    print(model.old_sigma)
    print(model.N_K)
    num_graph = model.M
    xx = np.linspace(0, np.pi*3, 100)[:,None]
    xx = torch.from_numpy(xx).float()
    mm, ss = model.predict(xx)

    mm = mm.numpy()
    ss = np.sqrt(ss.numpy())
    xx = xx.numpy().ravel()

    plt.scatter(X, Y1)
    plt.scatter(X, Y2)
    # plt.scatter(X3, Y3)
    for m in range(num_graph):
        line = plt.plot(xx, mm[m])
        plt.plot(xx, mm[m,:,0]+ss[m], "--", color=line[0].get_color())
        plt.plot(xx, mm[m,:,0]-ss[m], "--", color=line[0].get_color())

    plt.show()

