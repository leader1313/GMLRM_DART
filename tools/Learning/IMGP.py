import numpy as np
import torch
import copy

class IMGP:
    def __init__(self, X, Y, M, iters, kernel):
        self.X = X
        self.Y = Y

        self.N = self.X.shape[0]
        self.M = M
        self.D = self.Y.shape[1]
        self.T = iters
        self.prob_thresh = 0.001
        self.Noise = np.zeros(self.D)
        
        self.kern = np.array([kernel() for _ in range(self.M)])

        self.alpha = torch.tensor(np.log(1000)).float()
        self.log_p_y_sigma = torch.tensor(np.log(0.05)).float()

        self.q_z_pi = torch.ones(self.M, self.N) / self.M

        self.v_beta_a = torch.ones(self.M)
        self.v_beta_b = torch.ones(self.M)*torch.exp(self.alpha)

        self.q_f_mean = torch.tensor(np.random.normal(0, 0.01, (self.M, self.N, self.D))).float()
        self.q_f_sig = torch.stack([torch.eye(self.N) for m in range(self.M)])

    def update_q_z(self):
        sigma = torch.exp(self.log_p_y_sigma)
        E_ln_v = torch.digamma(self.v_beta_a) - torch.digamma(self.v_beta_a+self.v_beta_b)
        E_ln_1_minus_v = torch.digamma(self.v_beta_b) - torch.digamma(self.v_beta_a+self.v_beta_b)

        tmp_sum = torch.zeros(self.M)
        for m in range(0, self.M):
            tmp_sum[m] += E_ln_v[m]
            for i in np.arange(0, m):
                tmp_sum[m] += E_ln_1_minus_v[i]

        log_pi = -0.5/sigma * ((self.Y.repeat(self.M,1,1)-self.q_f_mean)**2).sum(2) \
                    -0.5/sigma * torch.stack([torch.diag(self.q_f_sig[m]) for m in range(self.M)]) \
                    -0.5*torch.log(np.pi*2*sigma)*self.D \
                    + (tmp_sum).repeat(self.N,1).T

        self.q_z_pi = torch.exp(log_pi)

        self.q_z_pi /= self.q_z_pi.sum(0)[None,:]
        
        self.q_z_pi[torch.isnan(self.q_z_pi)] = 1.0/self.M

        n_digits = 3
        
        self.q_z_pi = (self.q_z_pi * 10**n_digits).round() / (10**n_digits)

    def update_q_f(self):
        sigma = torch.exp(self.log_p_y_sigma)
        K = torch.stack([self.kern[m].K(self.X)+torch.eye(self.N)*1e-5 for m in range(self.M)])
        # invK = torch.inverse(K)
        sqrtK = torch.sqrt(K)
        B = torch.stack([torch.diag(self.q_z_pi[m]/sigma) for m in range(self.M)])

        # self.q_f_sig = torch.inverse(invK+B)
        self.q_f_sig = sqrtK.bmm(torch.solve(sqrtK, torch.stack([torch.eye(self.N) for _ in range(self.M)])+sqrtK.bmm(B).bmm(sqrtK))[0])
        self.q_f_mean = self.q_f_sig.bmm(B).bmm(self.Y.repeat(self.M,1,1))

    def update_q_v(self):
        alpha = torch.exp(self.alpha)
        for m in range(0,self.M):
            self.v_beta_a[m] = 1.0 + self.q_z_pi[m,:].sum()
            tmpSum = torch.zeros(1)
            for i in range(m+1,self.M):
                tmpSum += self.q_z_pi[i,:].sum()

            self.v_beta_b[m] = alpha + tmpSum

    def predict(self, x):
        sigma = torch.exp(self.log_p_y_sigma)
        K = torch.stack([self.kern[m].K(self.X)+torch.eye(self.N)*1e-5 for m in range(self.M)])

        B = torch.stack([torch.diag(self.q_z_pi[m]/sigma) for m in range(self.M)])
        sqrtB = torch.sqrt(B)
        R = torch.stack([torch.eye(self.N)+sqrtB[m].mm(K[m]).mm(sqrtB[m]) for m in range(self.M)])

        Kx = torch.stack([self.kern[m].K(x) for m in range(self.M)])
        Knx = torch.stack([self.kern[m].K(self.X, x) for m in range(self.M)])

        mean = Knx.permute(0,2,1).bmm(sqrtB).bmm(torch.solve(sqrtB, R)[0]).bmm(self.Y.repeat(self.M,1,1))
        sigma = torch.stack([torch.diag(Kx[m] - Knx[m].t().mm(sqrtB[m]).mm(torch.solve(sqrtB[m], R[m])[0]).mm(Knx[m]))+sigma for m in range(self.M)])
        sigma[sigma<0.00001]=0.00001
        return mean, sigma

    def compute_grad(self, flag):
        self.alpha.requires_grad = flag
        self.log_p_y_sigma.requires_grad = flag
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                self.kern[m].compute_grad(flag)

    def expectation(self, N=10):
        for i in range(N):
            self.update_q_f()
            self.update_q_v()
            self.update_q_z()
        
    def save_checkpoint(self):
        sigma = self.log_p_y_sigma
        alpha = self.alpha
        parameters = [sigma,alpha]
        q_z_pi = self.q_z_pi
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                theta = self.kern[m].param()
                parameters += theta
        torch.save({
                    'q_z_pi':q_z_pi
                    ,'parameters':parameters
                    }
                    ,'data/checkpoint.pt')

    def load_checkpoint(self):
        checkPoint = torch.load('data/checkpoint.pt')
        parameters = checkPoint['parameters']
        self.q_z_pi = checkPoint['q_z_pi']
        self.log_p_y_sigma = parameters[0]
        self.alpha = parameters[1]
        i = 0
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                self.kern[m].sigma = (parameters[i+2])
                i += 1

    def negative_lowerbound(self, n_batch=None):
        
        if n_batch is None:
            ind = torch.arange(self.N)
        else:
            ind = torch.randperm(self.N)[:n_batch]
        
        alpha = torch.exp(self.alpha)
        sigma = torch.exp(self.log_p_y_sigma)

#Expectation of likelihood        

        K = torch.stack([self.kern[m].K(self.X[ind])+torch.eye(ind.shape[0])*1e-5 for m in range(self.M)])
        q_z_pi = copy.deepcopy(self.q_z_pi)
        q_z_pi[q_z_pi!=0] /= sigma
        B = torch.stack([torch.diag(q_z_pi[m][ind]) for m in range(self.M)])
        sqrtB = torch.sqrt(B)

        R = torch.stack([torch.eye(ind.shape[0])+sqrtB[m].mm(K[m]).mm(sqrtB[m]) for m in range(self.M)])

        lk_1 = torch.zeros(1)
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                lk_1 += self.D*torch.slogdet(R[m])[1]
                lk_1 += 0.5*torch.trace(self.Y[ind].t().mm(sqrtB[m]).mm(torch.solve(sqrtB[m], R[m])[0]).mm(self.Y[ind]))

                lk_2 = 0.5*(self.q_z_pi[m]*torch.log(np.pi*2*sigma)).sum()

        E_zv = torch.zeros(1)
        kl_v = torch.zeros(1)

        v_beta_a = torch.ones(self.M)
        v_beta_b = torch.ones(self.M)*alpha

        num_M = 0
        for m in range(0,self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                num_M += 1
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
        for m in range(0,self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                tmp_sum[m] += E_ln_v[m]
                for i in range(0, m-1):
                    tmp_sum[m] += E_ln_1_minus_v[i]

                E_zv += (self.q_z_pi[m][ind]*(tmp_sum[m])).sum()
                # E_zv -= self.q_z_pi[m][ind][None,...].mm(torch.log(self.q_z_pi[m][ind][...,None])).sum()
                
                # ln_rho += -0.5/sigma * ((self.Y[ind]-self.q_f_mean[m][ind])**2).sum(1) \
                #             -0.5/sigma * torch.diag(self.q_f_sig[m])[ind] \
                #             -0.5*torch.log(np.pi*2*sigma)*self.D \
                #             + (tmp_sum[m])
        
        # for m in range(0,self.M):
        #     if self.q_z_pi[m].sum() > self.prob_thresh:
        #         ln_rho_1 = -0.5/sigma * ((self.Y[ind]-self.q_f_mean[m][ind])**2).sum(1) \
        #                     -0.5/sigma * torch.diag(self.q_f_sig[m])[ind] \
        #                     -0.5*torch.log(np.pi*2*sigma)*self.D

                # E_zv += self.q_z_pi[m][ind][None,...].mm((ln_rho_1 - ln_rho)[...,None]).sum()

#======================================================================================                

# KL(v*|v)
        for m in range(0,self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                kl_v += torch.lgamma(v_beta_a[m]+v_beta_b[m])-torch.lgamma(v_beta_a[m])-torch.lgamma(v_beta_b[m])
                kl_v += torch.lgamma(alpha)-torch.lgamma(1+alpha)
                kl_v += (v_beta_a[m]-1)*(E_ln_v[m])+(v_beta_b[m]-alpha)*(E_ln_1_minus_v[m])
                
        
        return (lk_1 + lk_2 - E_zv + kl_v)/self.N

    def learning(self, N=3):
        Max_step = self.T
        NL = self.negative_lowerbound()
        self.save_checkpoint()
        step = 0
        stop_flag = False
        Max_patient = 10
        patient_count = 0 
        while ((step < Max_step) and not(stop_flag)):
            step += 1
            print("=========================")
            print('E step')
            self.expectation(300)
            print('M step')
            self.maximization()
            
            print(step,' th NL : ',self.negative_lowerbound())
            print('Sigma : ',torch.exp(self.log_p_y_sigma))
            print('Alpha : ',torch.exp(self.alpha))
            print("Z : ",self.q_z_pi.sum(axis = 1))
            if NL > self.negative_lowerbound():
                patient_count = 0
                NL = self.negative_lowerbound()
                self.save_checkpoint()
                
            else : 
                patient_count += 1
                print("-------Patient_Count(< %i) : %i"%(Max_patient,patient_count))
                if patient_count >= Max_patient: 
                    stop_flag = True
                    
        self.load_checkpoint()
        limit_prob = self.N*(1/self.M)*0.1
        
        print(self.q_z_pi.sum(axis = 1))
        M = self.q_z_pi.sum(axis = 1) > limit_prob
        
        M = M.numpy()
        
        num_Mixture = M.sum()
            
        self.kern = self.kern[M]

        self.q_z_pi = self.q_z_pi[self.q_z_pi.sum(axis = 1) > limit_prob]
        self.M = num_Mixture
        self.Noise = np.exp(self.log_p_y_sigma.numpy())
        print('-------------------------------------------')
        print('Number of Mixture : %i'%(num_Mixture))
        print('Optimized Noise : %f'%(self.Noise))
        

    def maximization(self):
        max_iter = 100
        
        self.compute_grad(True)
        param = [self.log_p_y_sigma]
        param += [self.alpha]
        for m in range(self.M):
            if self.q_z_pi[m].sum() > self.prob_thresh:
                param += self.kern[m].param()
        optimizer = torch.optim.Adam(param,lr=0.01)
        # optimizer = torch.optim.Adadelta(param,lr=0.01)
        # optimizer = torch.optim.SGD(param,lr=0.1,momentum=0.9)
        print(optimizer.state_dict)
        for i in range(max_iter):
            optimizer.zero_grad()

            try:
                f = self.negative_lowerbound(n_batch=10)
                f.backward()
            except (TypeError, RuntimeError) as e:
                print(e)
                import ipdb; ipdb.set_trace()

            optimizer.step()

        self.compute_grad(False)
    
    

if __name__=="__main__":
    import sys
    from kernel import GaussianKernel
    import matplotlib.pyplot as plt
    plt.style.use("ggplot")

    N = 20
    X = np.linspace(0, -np.pi*2, N)[:,None]
    Y1 = np.sin(X) + np.random.randn(N)[:,None] * 0.1
    Y2 = np.cos(X) + np.random.randn(N)[:,None] * 0.1
    

    X = torch.from_numpy(X).float()
    Y1 = torch.from_numpy(Y1).float()
    Y2 = torch.from_numpy(Y2).float()

    kern = GaussianKernel()
    model = IMGP(torch.cat([X,X]).float(), torch.cat([Y1, Y2]).float(), 5,30, GaussianKernel)

    model.learning()
    
    num_graph = model.M
    xx = np.linspace(0, -np.pi*2, 100)[:,None]
    xx = torch.from_numpy(xx).float()
    mm, ss = model.predict(xx)

    mm = mm.numpy()
    ss = np.sqrt(ss.numpy())
    xx = xx.numpy().ravel()

    plt.scatter(X, Y1)
    plt.scatter(X, Y2)
    for m in range(num_graph):
        line = plt.plot(xx, mm[m])
        plt.plot(xx, mm[m,:,0]+ss[m], "--", color=line[0].get_color())
        plt.plot(xx, mm[m,:,0]-ss[m], "--", color=line[0].get_color())

    plt.show()

