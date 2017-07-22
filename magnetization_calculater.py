# -*- coding: utf-8 -*-
"""
Created on Sat Jul 22 13:47:27 2017

@author: lukas
"""

###############################################################################
#######################     Basic Imports           ###########################
###############################################################################

import numpy as np; np.set_printoptions(threshold = 50)
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from time import time

###############################################################################

###############################################################################
#######################     setting up pathes       ###########################
###############################################################################
#%%
"""
defining global pathes
1. directory of the script containing this function and hence the package!
2. directory of the index files
3. directory of the previously calculated magnetizations
"""
global package_path
global index_path
global mag_path
    
package_path = os.path.dirname(os.path.abspath(__file__))
index_path = os.path.join(package_path, "index_files")
mag_path = os.path.join(package_path, "mag_database")
#%%
###############################################################################

###############################################################################
#######################    skyrmion model import    ###########################
###############################################################################

import skyrmion_model_routines as smr

###############################################################################

###############################################################################
#######################        init variables       ###########################
############################################################################### # see page 5 

BC2 = 45.2919                                                                   # Bc2 as definded, without dipole interaction
Bx, By, Bz = 0., 0., BC2/2.                                                       # right now arbitrary values, in units of T
Bhom = np.array([Bx, By, Bz])
B = np.linalg.norm(Bhom)                                                        # external Bfield in e3 dir
dirNSky = Bhom/B

#------------------------------------------------------------------------------ # see page 6

nMax = 300
qMax = 3.1                                                                      # nMax=Anzahl moeglicher q-Vektoren, qMax=radius um Q=0 in dem alle betrachteten q-Vektoren liegen

#------------------------------------------------------------------------------ # see page 6

Q1Start = np.array([1.,0.,0.])
Q2Start = .5 * np.array([-1., np.sqrt(3), 0.])

#------------------------------------------------------------------------------

q1 = 1.
q2 = 0.
q3 = 0.

#------------------------------------------------------------------------------

Nx = 1./3
Ny = 1./3
Nz = 1./3

DemN = np.array([[Nx, 0., 0.], [0., Ny, 0.], [0., 0., Nz]])

DuD = 2 * 0.34
t = -1000

#------------------------------------------------------------------------------

qRoh = smr.loadqInd(qMax); nQ = len(qRoh) - 1
qRohErw = smr.loadqInd(qMax, 4.); nQErw = len(qRohErw) - 1

Q1, Q2 = smr.initQ(q1,q2, q3, dirNSky)

Q = np.array([smr.q(i, qRoh, qRohErw, Q1, Q2) for i in xrange(nQ+1)])

uel = smr.unique_entries(Q)

mag0real = smr.buildmag0(uel)

Ringe = 3

###############################################################################

for i in np.round([BC2/2.-BC2/32., BC2/2.+BC2/32.], 3):

    t0 = time()
    res = smr.groundState(q1, q2, q3, i * dirNSky, t, DuD, qRoh, qRohErw, mag0real, dirNSky, uel, Ringe)

    smr.reswriter(res.x, t, i, Ringe)

    print (time()-t0)/3600.


















# calc indices for Phi4Term2
"""
ipath = os.path.join(index_path, "4_Ringe_Phi4.ind")

f = open(ipath, "w")

for i1 in xrange(nQ+1):#-1):
    for i2 in xrange(nQ+1):
        for i3 in xrange(nQ+1):
            for i4 in xrange(nQ+1):
                if np.all(qRoh[i1] + qRoh[i2] + qRoh[i3] + qRoh[i4] == np.array([0, 0], dtype = np.int8)):
                    s = "%i, %i, %i, %i \n" % (i1, i2, i3, i4)
                    f.write(s)
                    
f.close()
"""