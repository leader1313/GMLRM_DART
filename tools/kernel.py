import torch
import numpy as np

class GaussianKernel:
  def __init__(self, param=None):
    if param is None:
      self.alpha = torch.tensor([np.log(1.0)])
      self.sigma = torch.tensor([np.log(1.0)])
    else:
      self.alpha = torch.tensor([np.log(param[0])])
      self.sigma = torch.tensor([np.log(param[1])])

  def K(self, x1, x2=None):
    if x2 is None:
      mu = x1.mean(0)
      a = x1 - mu.repeat(x1.shape[0], 1)
      b = (a**2).sum(1)[:,None]
      c = (b+b.t() - 2*a.mm(a.t())) / torch.exp(self.sigma)

    else:
      mu = torch.cat([x1, x2], 0).mean(0)
      a1 = x1 - mu.repeat(x1.shape[0], 1)
      a2 = x2 - mu.repeat(x2.shape[0], 1)
      b1 = (a1**2).sum(1)[:,None]
      b2 = (a2**2).sum(1)[:,None]
      c = (b1+b2.t() - 2*a1.mm(a2.t())) / torch.exp(self.sigma)

    return torch.exp(self.alpha) * torch.exp(-0.5*c)
    
    # if x2 is None:
    #   a = (x1 - x1.t())**2 / torch.exp(self.sigma)

    # else:
    #   a = (x1 - x2.t())**2 / torch.exp(self.sigma)

    # return torch.exp(self.alpha) * torch.exp(-0.5*a)

  def param(self):
    # return [self.alpha, self.sigma]
    return [self.sigma]

  def compute_grad(self, flag):
    # self.alpha.requires_grad = flag
    self.sigma.requires_grad = flag

  def param_set(self, X, Y):
    pass

class Matern52:
  def __init__(self, param=None):
    if param is None:
      self.alpha = torch.tensor([np.log(1.0)])
      self.sigma = torch.tensor([np.log(2.0)])
    else:
      self.alpha = torch.tensor([param[0]])
      self.sigma = torch.tensor([param[1]])

  def K(self, x1, x2=None):
    sig = torch.exp(self.sigma)

    if x2 is None:
      b = (x1 - x1.t())**2 / torch.exp(self.sigma)
      c = torch.sqrt(b)

    else:
      b = (x1 - x2.t())**2 / torch.exp(self.sigma)
      c = torch.sqrt(b)

    return (torch.exp(self.alpha)+0.01) * (1+np.sqrt(5/sig)*c + 5/3*b/sig) * torch.exp(-np.sqrt(5/sig)*c)

  def param(self):
    # return [self.alpha, self.sigma]
    return [self.sigma]

  def compute_grad(self, flag):
    # self.alpha.requires_grad = flag
    self.sigma.requires_grad = flag

  def param_set(self, X, Y):
    pass

class Matern32:
  def __init__(self, param=None):
    if param is None:
      self.alpha = torch.tensor([np.log(1.0)])
      self.sigma = torch.tensor([np.log(2.0)])
    else:
      self.alpha = torch.tensor([param[0]])
      self.sigma = torch.tensor([param[1]])

  def K(self, x1, x2=None):
    sig = torch.exp(self.sigma)

    if x2 is None:
      b = (x1 - x1.t())**2 / torch.exp(self.sigma)
      c = torch.sqrt(b)

    else:
      b = (x1 - x2.t())**2 / torch.exp(self.sigma)
      c = torch.sqrt(b)

    return (torch.exp(self.alpha)+0.01) * (1+np.sqrt(3/sig)*c) * torch.exp(-np.sqrt(3/sig)*c)

  def param(self):
    # return [self.alpha, self.sigma]
    return [self.sigma]

  def compute_grad(self, flag):
    # self.alpha.requires_grad = flag
    self.sigma.requires_grad = flag

  def param_set(self, X, Y):
    pass

