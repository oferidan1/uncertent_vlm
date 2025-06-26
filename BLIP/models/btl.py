import torch
import torch.nn as nn
from torch.distributions.normal import Normal

def negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin = 0.0):
    muA2 = muA**2
    muP2 = muP**2
    muN2 = muN**2
    varP2 = varP**2
    varN2 = varN**2

    mu = torch.sum(muP2 + varP - muN2 - varN - 2*muA*(muP - muN), dim=0)
    T1 = varP2 + 2*muP2 * varP + 2*(varA + muA2)*(varP + muP2) - 2*muA2 * muP2 - 4*muA*muP*varP
    T2 = varN2 + 2*muN2 * varN + 2*(varA + muA2)*(varN + muN2) - 2*muA2 * muN2 - 4*muA*muN*varN
    T3 = 4*muP*muN*varA
    sigma2 = torch.sum(2*T1 + 2*T2 - 2*T3, dim=0)
    sigma = sigma2**0.5

    probs = Normal(loc = mu, scale = sigma + 1e-8).cdf(margin)
    nll = -torch.log(probs + 1e-8)

    return nll.mean()

def kl_div_gauss(mu_q, var_q, mu_p, var_p):
    N, D = mu_q.shape

    # kl diverence for isotropic gaussian
    kl = 0.5 * ((var_q / var_p) * D + \
    1.0 / (var_p) * torch.sum(mu_p**2 + mu_q**2 - 2*mu_p*mu_q, axis=1) - D + \
    D*(torch.log(var_p) - torch.log(var_q)))

    return kl.mean()

def kl_div_vMF(mu_q, var_q):
    N, D = mu_q.shape

    # we are estimating the variance and not kappa in the network.
    # They are propertional
    kappa_q = 1.0 / var_q
    kl = kappa_q - D * torch.log(2.0)

    return kl.mean()

class BayesianTripletLoss(nn.Module):

    def __init__(self, margin, varPrior, kl_scale_factor = 1e-6, distribution ='gauss'):
        super(BayesianTripletLoss, self).__init__()

        self.margin = margin
        self.varPrior = varPrior
        self.kl_scale_factor = kl_scale_factor
        self.distribution = distribution

    def forward(self, x, label):

        # divide x into anchor, positive, negative based on labels
        D, N = x.shape
        nq = torch.sum(label.data == -1).item() # number of tuples
        S = x.size(1) // nq # number of images per tuple including query: 1+1+n
        A = x[:, label.data == -1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0)
        P = x[:, label.data == 1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0)
        N = x[:, label.data == 0]

        varA = A[-1:, :]
        varP = P[-1:, :]
        varN = N[-1:, :]

        muA = A[:-1, :]
        muP = P[:-1, :]
        muN = N[:-1, :]

        # calculate nll
        nll = negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=self.margin)

        # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
        if self.distribution == 'gauss':
            muPrior = torch.zeros_like(muA, requires_grad = False)
            varPrior = torch.ones_like(varA, requires_grad = False) * self.varPrior

            kl = (kl_div_gauss(muA, varA, muPrior, varPrior) + \
            kl_div_gauss(muP, varP, muPrior, varPrior) + \
            kl_div_gauss(muN, varN, muPrior, varPrior))

        elif self.distribution == 'vMF':
            kl = (kl_div_vMF(muA, varA) + \
            kl_div_vMF(muP, varP) + \
            kl_div_vMF(muN, varN))

        return nll + self.kl_scale_factor * kl

    def __repr__(self):
        return self.__class__._Name__ + '(' +'margin=' + '{:.4f}'.format(self.margin) + ')'

