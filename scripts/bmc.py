from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np


class BMC(object):
    '''
    Bayesian Model Combination (Monteith et al. 2011).
    A Hybrid Ensemble Approach to Star-galaxy Classification (Kim, Brunner & Carrasco Kind 2015).
    '''
    def __init__(self, q=4, ncomb=1000, nburn=200, ntop=10, icore=0):
        '''
        Constructor.
        
        Parameters
        ----------
        q: The number of combinations drawn each time from Dirichlet distribution.
        ncomb: The total number of ensembles.
        nburn: The number of iterations in the burn-in step.
        ntop: The side length of a rectangular SOM.
        icore: The number of cores (0 unless parallelized).
        '''
        self.q = q
        self.ncomb = ncomb
        self.nburn = nburn
        self.pfloor = 1e-300

    def get_log_likelihood(self, label, array, weight):

        y = label
        x = np.sum(array * weight, axis=1)
        error = np.abs(y - x)
        
        # Cromwell's rule:
        # I beseech you, in the bowels of Christ,
        # think it possible that you may be mistaken.
        # and also the fact that log of 0 diverges.
        error[error == 0] = self.pfloor
        error[error == 1] = 1 - self.pfloor
        lnlike = np.log(1 - error).sum()

        return lnlike

    def fit(self, X, y, create_cells=True):

        nxrows, nxcols = X.shape
        self.m = nxcols

        post_all = np.zeros(nxrows)

        # create folders
        weight_all = np.zeros(((self.ncomb - self.nburn) * self.q, self.m))
        
        alpha = np.ones(self.m)
        log_p_comb = np.zeros(self.q)

        for i in range(self.ncomb):
            weight = np.random.dirichlet(alpha, self.q)

            for j in range(self.q):
                w = self.get_log_likelihood(y, X, weight[j])
                log_p_comb[j] = w
                        
                if i >= self.nburn:
                    weight_all[(i - self.nburn) * self.q + j] = weight[j]

            best_weight = weight[log_p_comb.argmax()]
            alpha += best_weight

        a = np.dot(X, weight_all.T)

        self.post = np.sum(a, axis=1) / ((self.ncomb - self.nburn) * self.q)
        self.weight_all = weight_all
        
    def predict_proba(self, X):
        
        post_sum = np.zeros(len(X))
            
        w = self.weight_all
        a = np.dot(X, w.T)
        post = np.sum(a, axis=1) / len(w)
                
        return post
