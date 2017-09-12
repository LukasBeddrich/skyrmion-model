# -*- coding: utf-8 -*-

import skyrmion_model_routines as smr
import numpy as np
from multiprocessing import Pool
from time import time, sleep
from copy import deepcopy
#import pathos.pools as pp
#------------------------------------------------------------------------------
## Pathes



###############################################################################

""" !!! DOES IT WORK ??? """

def fluctuationM_newpool(kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD):
    """
    fluctuationM with multiprocessing Pool
    """
    
    nQloc = len(qRoh)
    fM = np.zeros((3*nQloc, 3*nQloc), dtype = np.complex)
    
    def g_n_poolprep(n):
        nn = n
        while nn <= (nQloc-1):
            subfM = np.asarray([[smr.g_ij2(n, nn, j, i, kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD) for i in (0,1,2)] for j in (0,1,2)], dtype = np.complex)
            fM[3*n:3*n+3, 3*nn:3*nn+3] = deepcopy(subfM)
            nn+=1
            
    p = Pool(processes=3)
    p.map(g_n_poolprep, range(nQloc))
    p.close()
    p.join()
    
    return 2. * (np.triu(fM) + np.conjugate(np.transpose(np.triu(fM,1))))

#------------------------------------------------------------------------------

def fluctuationM_new(kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD):
    """
    fluctuationM using hermitian symmertie
    """
        
    nQloc = len(qRoh)
    
    fM = np.zeros((3*nQloc, 3*nQloc), dtype = np.complex)
    
    
    for n in xrange(nQloc):
        nn = deepcopy(n)
        while nn <= (nQloc-1):
            subfM = np.asarray([[smr.g_ij2(n, nn, j, i, kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD) for i in (0,1,2)] for j in (0,1,2)], dtype = np.complex)
            fM[3*n:3*n+3, 3*nn:3*nn+3] = deepcopy(subfM)
            nn += 1
    

    return 2.*(np.triu(fM) + np.conjugate(np.transpose(np.triu(fM,1))))

###############################################################################

def pooling_fs(n):
    p = Pool(processes=2)
    arr = [n for n in xrange(n)]
    result = p.map(f3, arr)
    p.close()
    p.join()
    return result

def f1(x):
    sleep(1)
    return x*x

def f2(x):
    sleep(.5)
    return sum(x)

def f3(idxs):
    return np.zeros((3,3)) + idxs

###############################################################################

from pathos.pools import ParallelPool


def addup(x,y):
    sleep(10.)
    return x+y

if __name__ == '__main__':
    p = ParallelPool()
    inX = range(10)
    inY = range(10)
    
    print p.map(addup, inX, inY)
    p.close()
    p.join()





"""
The appropriate methods seem to be PROCESSES, not pools
"""