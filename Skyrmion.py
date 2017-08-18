#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 10:49:02 2017

@author: lbeddric
"""

###############################################################################
#######################                             ###########################
#######################                             ###########################
#######################    SKYRMION-MODEL 0.2.0     ###########################
#######################                             ###########################
#######################                             ###########################
###############################################################################

""" Initializes the necessary variables for Energy and Weights"""
""" Later on: rewrite in a more object oriented way"""

###############################################################################
#######################     Basic Imports           ###########################
###############################################################################
#%%
import numpy as np; np.set_printoptions(threshold = 50, precision = 15)
import os
import skyrmion_model_routines as smr

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
#######################        init variables       ###########################
############################################################################### # see page 5 
#%%
BC2 = 14.2467                                                                   # Bc2 for T = -100  as definded, without dipole interaction
Bfrac = 0.5
Bx, By, Bz = 0., 0., BC2*Bfrac                                                       # right now arbitrary values, in units of T
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

DuD = 2 * 0.34                                                                  # Dipole interaction strength for  >> MnSi <<
t = -5

#------------------------------------------------------------------------------

qRoh = smr.loadqInd(qMax); nQ = len(qRoh) - 1
qRohErw = smr.loadqInd(qMax, 4.); nQErw = len(qRohErw) - 1

Q1, Q2 = smr.initQ(q1,q2, q3, dirNSky)

Q = np.array([smr.q(i, qRoh, qRohErw, Q1, Q2) for i in xrange(nQ+1)])

uel = smr.unique_entries(Q)

mag0real = smr.buildmag0(uel)                                                       # keine komplexen zahlen! weniger speicher und ansonsten keine kompatibilität mit MINIMIZE
#mag = initmarray2(uel, mag0, qRoh, qRohErw, Q1, Q2)

#magmaticapath = os.path.join(mag_path, "magmatica_R_3.out")
#q1g, q2g, q3g, = np.genfromtxt(magmaticapath, delimiter = ",")[0]
#magmatica = np.genfromtxt(magmaticapath, delimiter = ",")[1:]

(q1g, q2g, q3g), magmatica = smr.magLoader(0.5, -100, 3, True)

Q1g, Q2g = smr.initQ(q1g, q2g, q3g, dirNSky)
Qg = np.array([smr.q(i, qRoh, qRohErw, Q1g, Q2g) for i in xrange(nQ+1)])
m = smr.initmarray(uel, smr.magtoimag(magmatica), Qg)

###############################################################################

###############################################################################
#######################        calculate Disp       ###########################
###############################################################################

#Borient = np.array([0,0,1])
#NuclearBragg = np.array([1,1,0])
#QVector = np.array([1.,1.,0.])

def disp_skyrmion(Borient, NuclearBragg, QVector, Kvector):
    """
    
    """
    try:
        eEnergies, weights = smr.select_EW_from_table(BC2, t, int(qMax), Bfrac, Borient, NuclearBragg, QVector, 0.01, Kvector, 0.01).T
        print 'Found values in database!'
    except ValueError:
        print 'No entries found in database. Calculation started!'
        eEnergies, weights = smr.EnergyWeightsMagnonsFalt(m, qRoh, qRohErw, Qg, q1g, q2g, q3g, t, DuD, Borient, NuclearBragg, QVector, Kvector).T
        smr.add_EW_to_table(BC2, t, int(qMax), Bfrac, Borient, NuclearBragg, QVector, Kvector, eEnergies, weights)
    return eEnergies, weights

###############################################################################
