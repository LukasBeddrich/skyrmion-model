# -*- coding: utf-8 -*-

import skyrmion_model_routines as smr
import numpy as np
import multiprocessing
from time import time, sleep
from copy import deepcopy
import sys
#import pathos.pools as pp
#------------------------------------------------------------------------------
## Pathes



###############################################################################

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
"""
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

np.random.seed(0)
sample = np.random.rand(12)
neworder = np.random.randint(12)
"""
"""
def worker():
    name = multiprocessing.current_process().name
    print name, 'Starting'
    sleep(2)
    print name, 'Exiting'
    
def my_service():
    name = multiprocessing.current_process().name
    print name, 'Starting'
    sleep(3)
    print name, 'Exiting'
"""
"""
class Consumer(multiprocessing.Process):
    
    def __init__(self, task_queue, result_queue):
        multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue
        
    def run(self):
        proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print '%s: Exiting' % proc_name
                self.task_queue.task_done()
                break
            print '%s: %s' % (proc_name, next_task)
            answer = next_task()
            self.task_queue.task_done()
            self.result_queue.put(answer)
        return            

class Task(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def __call__(self):
        sleep(0.1)
        return '%s * %s = %s' % (self.a, self.b, self.a * self.b)
    def __str__(self):
        return '%s * %s' % (self.a, self.b)
    
    
if __name__ == '__main__':
    d = multiprocessing.Process(name='daemon', target=daemon)
    d.daemon = True

    n = multiprocessing.Process(name='non-daemon', target=non_daemon)
    n.daemon = False

    d.start()
    sleep(1)
    n.start()
    
    d.join(1)
    print 'd.is_alive()', d.is_alive()
    n.join()
"""

def calc_square(numbers, result):
    pass
"""
The appropriate methods seem to be PROCESSES, not pools
"""