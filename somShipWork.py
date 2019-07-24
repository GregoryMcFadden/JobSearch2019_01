
# # Lesson 10 - Self Organizing Maps Part 2


import math
import numpy as np
import sys

class SOM(object):
    def __init__(self, n, r, c=None):
        self.n = n
        self.r = r  # rows in the lattice
        self.c = r if c is None else c  # columns in the lattice
        self.w = np.random.random([self.r, self.c, self.n])  # initialize with random values
        #print (self.w)

    def find_bmu(self, inp):
        best_d = float("inf")
        best_r = -1
        best_c = -1
        for r in range(self.r):
            for c in range(self.c):
                d = (np.linalg.norm(inp - self.w[r, c]))**2  # calculate distance squared
                if d < best_d:  # keep best (smallest) distance squared, best row and best column
                    best_d = d
                    best_r = r
                    best_c = c
                    
        return best_r, best_c

    def train(self, data, alpha0=0.8, sigma0=None, lam=None, t_max=100):
        if sigma0 is None:
            # No sigma0 was provided, take 1/2 of width or height lattice, whichever is bigger
            sigma0 = max(self.r, self.c) * 0.5
        if lam is None:
            # No lambda was provided, calculate one.
            lam = t_max / math.log(sigma0)
        t = 0
        while t < t_max:  # stop if we did all the iterations
            if not (t+1)%10:
                print("Iteration " + str(t+1) + " / " + str(t_max) )
            factor = math.exp(-t / lam)  # calculate exponential decay factor
            sigma = sigma0 * factor
            #print("sigma " + str(sigma))
            if sigma < 1.0:  # stop if neighborhood is too small
                break
            alpha = alpha0 * factor
            inp = data[np.random.randint(data.shape[0])]  # pick a random input
            bmu = self.find_bmu(inp)
            self.update_weights(bmu, inp, alpha, sigma)
            t += 1


# Updating the weights around a BMU:

# In[4]:


    def update_weights(self, bmu, inp, alpha, sigma):
        
        sigma2 = sigma * sigma
        deno = -2.0 * sigma2

        bmu_r, bmu_c = bmu
        for r in range(self.r):
            dr = (r - bmu_r)**2
            for c in range(self.c):
                dc = (c - bmu_c)**2
                d2 = dr + dc
                # Check if the (r,c) in the neighborhood
                if d2 < sigma2:
                    # calculate the influence of (r,c)
                    theta = np.exp(d2 / deno)
                    # update the weight at that (r,c)
                    self.w[r, c] += alpha * theta * (inp[0] - self.w[r, c])
