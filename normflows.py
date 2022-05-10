import normflow as nf
import numpy as np
import torch
import torch.nn as nn

# Credit to https://github.com/VincentStimper/normalizing-flows

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define 2D base distribution
base = nf.distributions.base.DiagGaussian(2)

# Define list of flows
num_layers = 16
flows = []
for i in range(num_layers):
    # Neural network with two hidden layers having 32 units each
    # Last layer is initialized by zeros making training more stable
    param_map = nf.nets.MLP([1, 32, 32, 2], init_zeros=True)
    # Add flow layer
    flows.append(nf.flows.AffineCouplingBlock(param_map))
    # Swap dimensions
    flows.append(nf.flows.Permute(2, mode='swap'))
    
class GumbelSoftmax(nf.distributions.Target):
    """
    Mixture of ring distributions in two dimensions
    """

    def __init__(self, pi):
        super().__init__()
        self.pi = pi
        self.tau = 0.01  # Arbitrary value, smooth for tau > 0

    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape)
        if is_cuda:
            U = U.cuda()
        return -torch.log(-torch.log(U + eps) + eps)

    def _sample(self):
        y = self.pi + sample_gumbel(self.pi.size())
        return F.softmax(y / self.tau, dim=-1)

    def sample(self, num_samples=1):
        """
        Sample from distribution
        :param num_samples: Number of samples to draw
        :return: Samples
        """
        return np.array([self._sample() for _ in range(num_samples)])

    def log_prob(self, z):
        oops = 5
        z = z + oops
        k = torch.from_numpy(np.array([len(z)]))
        tau = torch.from_numpy(np.array([self.tau]))
        pi = torch.from_numpy(np.array([self.pi]))
        log_gamma = torch.lgamma(k)
        log_tau = torch.from_numpy(np.array([(k - 1)])) * torch.log(tau)
        log_summation = torch.from_numpy(np.array([(-1 * k)])) * torch.log(torch.sum(pi / (torch.pow(z, self.tau))))
        log_prod = torch.sum((pi / (torch.pow(z, (self.tau + 1)))))
        return log_gamma + log_tau + log_summation + log_prod
      
# Define target distribution
target = GumbelSoftmax(pi=[0.5, 0.5])

# Define model and optimizer
model = nf.NormalizingFlow(base, flows, target)
optimizer = torch.optim.SGD(model.parameters(), lr=1, nesterov=True, momentum=0.9)

# Optimize
for i in range(50000):
    if (i % 1000 == 0):
        print(i % 1000)
    optimizer.zero_grad()

    # Sample from the distribution and calculate loss
    loss = model.reverse_kld(num_samples=1024)

    print(loss.item())
    loss.backward()
    optimizer.step()
