import numpy as np
import torch
import copy

class OMGP:
    def __init__(self, X, Y, M, T, kernel):
        self.X = X
        self.Y = Y

        self.N = self.X.shape[0]
        self.M = M
        self.D = self.Y.shape[1]
        self.T = T
        self.Noise = np.zeros(self.D)

        self.kern = [kernel() for _ in range(self.M)]

        self.log_p_y_sigma = torch.tensor(np.log(0.2)).float()

        self.p_z_pi = torch.ones(self.M, self.N) / self.M
        self.q_z_pi = torch.ones(self.M, self.N) / self.M

        self.q_f_mean = torch.tensor(np.random.normal(0, 0.01, (self.M, self.N, self.D))).float()
        self.q_f_sig = torch.stack([torch.eye(self.N) for m in range(self.M)])


    def update_q_z(self):
        sigma = torch.exp(self.log_p_y_sigma)
        log_pi = -0.5/sigma * ((self.Y.repeat(self.M,1,1)-self.q_f_mean)**2).sum(2) \
                -0.5/sigma * torch.stack([torch.diag(self.q_f_sig[m]) for m in range(self.M)]) \
                -0.5*torch.log(np.pi*2*sigma)*self.D
        
        self.q_z_pi = self.p_z_pi * torch.exp(log_pi)
    
        self.q_z_pi /= self.q_z_pi.sum(0)[None,:]
        self.q_z_pi[torch.isnan(self.q_z_pi)] = 1.0/self.M


    def update_q_f(self):
        sigma = torch.exp(self.log_p_y_sigma)
        K = torch.stack([self.kern[m].K(self.X)+torch.eye(self.N)*1e-5 for m in range(self.M)])
        # invK = torch.inverse(K)
        sqrtK = torch.sqrt(K)
        B = torch.stack([torch.diag(self.q_z_pi[m]/sigma) for m in range(self.M)])

        # self.q_f_sig = torch.inverse(invK+B)
        self.q_f_sig = sqrtK.bmm(torch.solve(sqrtK, torch.stack([torch.eye(self.N) for _ in range(self.M)])+sqrtK.bmm(B).bmm(sqrtK))[0])
        self.q_f_mean = self.q_f_sig.bmm(B).bmm(self.Y.repeat(self.M,1,1))


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


    def save_checkpoint(self):
        sigma = self.log_p_y_sigma
        parameters = [sigma]
        for m in range(self.M):
                theta = self.kern[m].param()
                parameters += theta
        torch.save(parameters,'data/checkpoint.pt')

    def load_checkpoint(self):
        param = torch.load('data/checkpoint.pt')

        return param

    def compute_grad(self, flag):
        self.log_p_y_sigma.requires_grad = flag
        for m in range(self.M):
            self.kern[m].compute_grad(flag)

    def expectation(self, N=100):
        for i in range(N):
            self.update_q_f()
            self.update_q_z()
  

    def negative_log_likelihood(self, n_batch=None):
        if n_batch is None:
            ind = torch.arange(self.N)
        else:
            ind = torch.randperm(self.N)[:n_batch]

        sigma = torch.exp(self.log_p_y_sigma)
        K = torch.stack([self.kern[m].K(self.X[ind])+torch.eye(ind.shape[0])*1e-5 for m in range(self.M)])
        q_z_pi = copy.deepcopy(self.q_z_pi)
        q_z_pi[q_z_pi!=0] /= sigma
        B = torch.stack([torch.diag(q_z_pi[m][ind]) for m in range(self.M)])
        sqrtB = torch.sqrt(B)

        R = torch.stack([torch.eye(ind.shape[0])+sqrtB[m].mm(K[m]).mm(sqrtB[m]) for m in range(self.M)])

        lk_y = torch.zeros(1)
        for m in range(self.M):
            lk_y += self.D*torch.slogdet(R[m])[1]
            lk_y += 0.5*torch.trace(self.Y[ind].t().mm(sqrtB[m]).mm(torch.solve(sqrtB[m], R[m])[0]).mm(self.Y[ind]))

        lk_z = 0.5*(self.q_z_pi*torch.log(np.pi*2*sigma)).sum()


        return (lk_y + lk_z)

  

    def learning(self, N=3):
        Max_step = self.T
        NL = self.negative_log_likelihood()
        self.save_checkpoint()
        step = 0
        stop_flag = False
        Max_patient = 5
        patient_count = 0 
        while ((step < Max_step) and not(stop_flag)):
            step += 1
            print("=========================")
            print('E step')
            self.expectation(300)
            print('M step')
            self.maximization()
            
            print(step,' th NL : ',self.negative_log_likelihood())
            print('Sigma : ',torch.exp(self.log_p_y_sigma))
            print("Z : ",self.q_z_pi.sum(axis = 1))
      

            if NL > self.negative_log_likelihood():
                patient_count = 0
                NL = self.negative_log_likelihood()
                self.save_checkpoint()
                
            else : 
                patient_count += 1
                print("-------Patient_Count(< %i) : %i"%(Max_patient,patient_count))
                if patient_count >= Max_patient:
                    parameters = self.load_checkpoint()
                    self.log_p_y_sigma = parameters[0]
                    for m in range(self.M):
                        self.kern[m].sigma = parameters[m+1]
                    stop_flag = True

        self.Noise = np.exp(self.log_p_y_sigma.numpy())
    


    def maximization(self):
        max_iter = 100

        self.compute_grad(True)
        param = [self.log_p_y_sigma]
        for m in range(self.M):
            param += self.kern[m].param()

        optimizer = torch.optim.Adam(param)

        for i in range(max_iter):
            optimizer.zero_grad()

        f = self.negative_log_likelihood(n_batch=10) 
        f.backward()
        optimizer.step()
        
        if torch.isnan(param[0]).sum()>0:
            import ipdb; ipdb.set_trace()
        

        self.compute_grad(False)
    

if __name__=="__main__":
  import sys
  from kernel import GaussianKernel
  import matplotlib.pyplot as plt
  plt.style.use("ggplot")

  N = 10
  M = 2
  X = np.linspace(0, np.pi*2, N)[:,None]
  Y1 = np.sin(X) + np.random.randn(N)[:,None] * 0.5
  Y2 = np.cos(X) + np.random.randn(N)[:,None] * 0.5

  X = torch.from_numpy(X).float()
  Y1 = torch.from_numpy(Y1).float()
  Y2 = torch.from_numpy(Y2).float()

  kern = GaussianKernel()
  model = OMGP(torch.cat([X,X]).float(), torch.cat([Y1, Y2]).float(), M,30, GaussianKernel)

  # for i in range(20):
  model.learning()
  
  xx = np.linspace(0, np.pi*2, 100)[:,None]
  xx = torch.from_numpy(xx).float()
  mm, ss = model.predict(xx)

  mm = mm.numpy()
  ss = np.sqrt(ss.numpy())
  xx = xx.numpy().ravel()

  plt.scatter(X, Y1)
  plt.scatter(X, Y2)
  for m in range(M):
    line = plt.plot(xx, mm[m])
    plt.plot(xx, mm[m,:,0]+ss[m], "--", color=line[0].get_color())
    plt.plot(xx, mm[m,:,0]-ss[m], "--", color=line[0].get_color())
  # if i%2 : plt.savefig('image/'+str(i))
  
  # plt.cla()
  plt.show()

