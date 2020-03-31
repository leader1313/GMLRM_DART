import numpy as np
import torch

class GPRegression:
  def __init__(self, X, Y, kernel):
    self.X = X
    self.Y = Y
    self.kern = kernel
    # self.sigma = torch.tensor([1.0], requires_grad=True)
    self.sigma = torch.tensor([np.log(0.3)])


  def predict(self, x):
    Kx = self.kern.K(x)
    Kxx = self.kern.K(self.X, x)
    K = self.kern.K(self.X)

    sig = torch.exp(self.sigma)

    mean = Kxx.t().mm(torch.solve(self.Y, K+torch.eye(K.shape[0])*sig)[0])
    sigma = torch.diag(Kx - Kxx.t().mm(torch.solve(Kxx, K+torch.eye(K.shape[0])*sig)[0])).reshape(x.shape[0], -1) + sig

    return mean, sigma

  def intended_action(self, state):
    intended_action, _ =self.predict(state)
    return intended_action

  def sample_action(self, state):
    mm, ss =self.predict(state)
    sample_action = ss * torch.randn(1) + mm
    return sample_action

  def compute_grad(self, flag):
    self.sigma.requires_grad = flag
    self.kern.compute_grad(flag)

  def negative_log_likelihood(self):
    K = self.kern.K(self.X) + torch.eye(self.X.shape[0])*torch.exp(self.sigma)

    self.K = K

    invKY = torch.solve(self.Y, K+torch.eye(self.Y.shape[0])*0.000001)[0]
    # logdet = torch.cholesky(K+torch.eye(K.shape[0])*0.000001, upper=False).diag().log().sum()
    sign, logdet = torch.slogdet(K+torch.eye(K.shape[0])*1e-6)
    return (logdet + self.Y.t().mm(invKY))

  def learning(self):
    max_iter = 3000
    # max_iter = 50

    self.compute_grad(True)
    param = self.kern.param() + [self.sigma]

    optimizer = torch.optim.Adam(param)

    for i in range(max_iter):
      optimizer.zero_grad()
      f = self.negative_log_likelihood() 
      f.backward()
      # def closure():
      #   optimizer.zero_grad()
      #   f = self.negative_log_likelihood() 
      #   f.backward()
      #   return f
      # optimizer.step(closure)
      optimizer.step()
    self.compute_grad(False)
    print('params:', torch.exp(self.kern.param()[0]), torch.exp(self.sigma))
    