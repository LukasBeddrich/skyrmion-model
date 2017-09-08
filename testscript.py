# -*- coding: utf-8 -*-

import skyrmion_model_routines as smr
import numpy as np
from multiprocessing import Pool
from time import time, sleep

#------------------------------------------------------------------------------

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

if __name__ == '__main__':
    res = np.array(np.reshape(pooling_fs(9),(9,9)))
    print res
    
"""
The appropriate methods seem to be PROCESSES, not pools
"""