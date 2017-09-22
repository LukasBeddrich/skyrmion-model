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
from scipy.linalg import eigvals, eig, inv
from copy import deepcopy

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
BC2 = 45.2919                                                                   # Bc2 for T = -100  as definded, without dipole interaction
Bfrac = 0.5
Bx, By, Bz = 0., 0., BC2*Bfrac                                                       # right now arbitrary values, in units of T
Bhom = np.array([Bx, By, Bz])
B = np.linalg.norm(Bhom)                                                        # external Bfield in e3 dir
dirNSky = Bhom/B

#------------------------------------------------------------------------------ # see page 6

nMax = 300
qMax = 7.1                                                                      # nMax=Anzahl moeglicher q-Vektoren, qMax=radius um Q=0 in dem alle betrachteten q-Vektoren liegen

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
t = -1000

#------------------------------------------------------------------------------

qRoh = smr.loadqInd(qMax); nQ = len(qRoh) - 1
qRohErw = smr.loadqInd(qMax, 4.); nQErw = len(qRohErw) - 1

Q1, Q2 = smr.initQ(q1,q2, q3, dirNSky)

Q = np.array([smr.q(i, qRoh, qRohErw, Q1, Q2) for i in xrange(nQ+1)])

uel = smr.unique_entries(Q)

#mag0real = smr.buildmag0(uel)                                                       # keine komplexen zahlen! weniger speicher und ansonsten keine kompatibilität mit MINIMIZE
#mag = initmarray2(uel, mag0, qRoh, qRohErw, Q1, Q2)

#magmaticapath = os.path.join(mag_path, "magmatica_R_3.out")
#q1g, q2g, q3g, = np.genfromtxt(magmaticapath, delimiter = ",")[0]
#magmatica = np.genfromtxt(magmaticapath, delimiter = ",")[1:]

(q1g, q2g, q3g), magmatica = smr.magLoader(Bfrac, t, int(qMax), True)

Q1g, Q2g = smr.initQ(q1g, q2g, q3g, dirNSky)
Qg = np.array([smr.q(i, qRoh, qRohErw, Q1g, Q2g) for i in xrange(nQ+1)])
m = smr.initmarray(uel, smr.magtoimag(magmatica), Qg)
print "Q1Start = " + str(Qg[3]) + "\nt = %f\nB = %f\nBfrac = %f" % (t, BC2, Bfrac)
###############################################################################

###############################################################################
#######################        calculate Disp       ###########################
###############################################################################

#Borient = np.array([0,0,1])
#NuclearBragg = np.array([1,1,0])
#QVector = np.array([1.,1.,0.])

def disp_skyrmion(Borient, NuclearBragg, QVector, Kvector, dbase = True):
    """
    Derivate of EnergyWeightsMagnonsFalt
    Calculates eigenenergies and weights without redundant computations!
    Borient [rlu]
    NuclearBragg [rlu]
    QVector [rlu]
    Kvector [k_h]
    """
    if dbase:
        
        try:
            print 'BC2 = {} \nT = {} \nrings = {} \nBfrac = {} \nBorient = {} \nNuclearbragg = {} \nQVector = {} \n\
            Kvector = {}'.format(BC2, t, int(qMax), Bfrac, Borient, NuclearBragg, QVector, Kvector)
            eEnergies, weights = smr.select_EW_from_table(BC2, t, int(qMax), Bfrac, Borient, NuclearBragg, QVector, 0.005, Kvector, 0.005).T

        except ValueError:
            print 'No entries found!\n'
            RotB = smr.find_rot_mat(Borient, np.asarray([0.,0.,1.]))
    
    # Additionally, since cubic anisotropies are neglected, the Q - structure points in an arbitrary direction. 
    # One now needs to rotate the system again to match a "real" Q - vector direction with the program - internal Q - direction, i.e. q[1]
            QInternal = Qg[1]
            nQInternal = np.linalg.norm(Qg[1])
            RotQ = smr.find_rot_mat(np.dot(RotB, QVector), QInternal)
    
    # ogether this forms the rotation matrix, that translates Vectors of the "real world" into program internal vectors
            RotMat = np.dot(RotQ, RotB)
            
    # To make life easier, the entered Kvector is normalized to units of QVectors
            Kvec = Kvector * nQInternal
            KvecRotated = np.dot(RotMat, Kvec)
            print 'KvecRotated = {}'.format(KvecRotated)
            
    # The nuclear Bragg vector in this coordinate system is then given by
            NuclearBraggRotated = np.dot(RotMat, NuclearBragg/np.linalg.norm(NuclearBragg))
            NuclearBraggRotated /= np.linalg.norm(NuclearBraggRotated)
            print "Lab system transformed into theory system"
    
            NM = np.zeros((3*len(qRohErw), 3*len(qRohErw)))
    
    # normalized nuclear Bragg vector
            NormGVector = np.concatenate((NuclearBraggRotated, np.zeros(3*(len(qRohErw) - 1))))
        
    # calculate EnergyspectrumFalt and save intermediate results
            MxF = smr.mCrossMatrixFalt(m, qRoh, qRohErw, np.asarray([KvecRotated[0], KvecRotated[1], KvecRotated[2]]), Qg[3], Qg[1]) # Qg's true?
#    return MxF
            SEV = smr.SelectedEigenvectors(MxF)
            MxSF = smr.chop(np.dot(np.dot(np.conjugate(np.transpose(SEV)), MxF),SEV))
            fMF = smr.fluctuationMFalt(KvecRotated[0], KvecRotated[1], KvecRotated[2], qRoh, qRohErw, m, Qg, q1g, q2g, q3g, t, DuD)
            chiI0SF = smr.chop(np.dot(np.dot(np.conjugate(np.transpose(SEV)), fMF),SEV))
            ESF = np.real(1.j * eigvals(np.dot(MxSF,chiI0SF)))
            ESF = np.sort(ESF[np.where(ESF > 0)[0]])
            
            MxSFinv = inv(MxSF)
            WeightEs = lambda i: eig( 1.j * ESF[i] *  MxSFinv + chiI0SF)
    
    # construct projection matrix that projects onto the first Brillouin zone and to the direction orthogonal to nuclear Bragg vector
            ProjMatrix = deepcopy(NM)
            ProjMatrix[:3,:3] = np.eye(3)
            ProjMatrix -= np.outer(NormGVector, NormGVector)
    
            EnergyWeight = []
            for i in xrange(len(ESF)):
                WeightVal, WeightVec = smr.chop2(WeightEs(i))
                inds = np.argsort(np.abs(WeightVal))[::-1]
                WeightVal, WeightVec = WeightVal[inds], WeightVec[:,inds]
                if WeightVal[-1] < 0.001:
                    tempw = np.real(np.trace(np.dot(ProjMatrix, np.dot(np.dot(SEV, np.dot(np.outer(WeightVec[:,-1], np.conjugate(WeightVec[:,-1])), 1.j * MxSF)), np.conjugate(np.transpose(SEV))))))
                    EnergyWeight.append([ESF[i], tempw])
                else:
                    print "Error!"
                    break
            eEnergies, weights = np.array(EnergyWeight).T
            smr.add_EW_to_table(BC2, t, int(qMax), Bfrac, Borient, NuclearBragg, QVector, Kvector, eEnergies, weights)
    
    else:
        RotB = smr.find_rot_mat(Borient, np.asarray([0.,0.,1.]))
    
    # Additionally, since cubic anisotropies are neglected, the Q - structure points in an arbitrary direction. 
    # One now needs to rotate the system again to match a "real" Q - vector direction with the program - internal Q - direction, i.e. q[1]
        QInternal = Qg[1]
        nQInternal = np.linalg.norm(Qg[1])
        RotQ = smr.find_rot_mat(np.dot(RotB, QVector), QInternal)
    
    # ogether this forms the rotation matrix, that translates Vectors of the "real world" into program internal vectors
        RotMat = np.dot(RotQ, RotB)
            
    # To make life easier, the entered Kvector is normalized to units of QVectors
        Kvec = Kvector * nQInternal
        KvecRotated = np.dot(RotMat, Kvec)
        print 'KvecRotated = {}'.format(KvecRotated)
            
    # The nuclear Bragg vector in this coordinate system is then given by
        NuclearBraggRotated = np.dot(RotMat, NuclearBragg/np.linalg.norm(NuclearBragg))
        NuclearBraggRotated /= np.linalg.norm(NuclearBraggRotated)
        print "Lab system transformed into theory system"
    
        NM = np.zeros((3*len(qRohErw), 3*len(qRohErw)))
    
    # normalized nuclear Bragg vector
        NormGVector = np.concatenate((NuclearBraggRotated, np.zeros(3*(len(qRohErw) - 1))))
        
    # calculate EnergyspectrumFalt and save intermediate results
        MxF = smr.mCrossMatrixFalt(m, qRoh, qRohErw, np.asarray([KvecRotated[0], KvecRotated[1], KvecRotated[2]]), Qg[3], Qg[1]) # Qg's true?
    #    return MxF
        SEV = smr.SelectedEigenvectors(MxF)
        MxSF = smr.chop(np.dot(np.dot(np.conjugate(np.transpose(SEV)), MxF),SEV))
        fMF = smr.fluctuationMFalt(KvecRotated[0], KvecRotated[1], KvecRotated[2], qRoh, qRohErw, m, Qg, q1g, q2g, q3g, t, DuD)
        chiI0SF = smr.chop(np.dot(np.dot(np.conjugate(np.transpose(SEV)), fMF),SEV))
        ESF = np.real(1.j * eigvals(np.dot(MxSF,chiI0SF)))
        ESF = np.sort(ESF[np.where(ESF > 0)[0]])
            
        MxSFinv = inv(MxSF)
        WeightEs = lambda i: eig( 1.j * ESF[i] *  MxSFinv + chiI0SF)
    
    # construct projection matrix that projects onto the first Brillouin zone and to the direction orthogonal to nuclear Bragg vector
        ProjMatrix = deepcopy(NM)
        ProjMatrix[:3,:3] = np.eye(3)
        ProjMatrix -= np.outer(NormGVector, NormGVector)
    
        EnergyWeight = []
        for i in xrange(len(ESF)):
            WeightVal, WeightVec = smr.chop2(WeightEs(i))
            inds = np.argsort(np.abs(WeightVal))[::-1]
            WeightVal, WeightVec = WeightVal[inds], WeightVec[:,inds]
            if WeightVal[-1] < 0.001:
                tempw = np.real(np.trace(np.dot(ProjMatrix, np.dot(np.dot(SEV, np.dot(np.outer(WeightVec[:,-1], np.conjugate(WeightVec[:,-1])), 1.j * MxSF)), np.conjugate(np.transpose(SEV))))))
                EnergyWeight.append([ESF[i], tempw])
            else:
                print "Error!"
                break
        eEnergies, weights = np.array(EnergyWeight).T
        smr.add_EW_to_table(BC2, t, int(qMax), Bfrac, Borient, NuclearBragg, QVector, Kvector, eEnergies, weights)
    return eEnergies/BC2, weights
#------------------------------------------------------------------------------    

def disp_skyrmion_old(Borient, NuclearBragg, QVector, Kvector):
    """
    
    """
    try:
        print 'BC2 = {} \nT = {} \nrings = {} \nBfrac = {} \nBorient = {} \nNuclearbragg = {} \nQVector = {} \n\
Kvector = {}'.format(BC2, t, int(qMax), Bfrac, Borient, NuclearBragg, QVector, Kvector)
        eEnergies, weights = smr.select_EW_from_table(BC2, t, int(qMax), Bfrac, Borient, NuclearBragg, QVector, 0.01, Kvector, 0.01).T
#        print 'Found values in database!'
    except ValueError:
        print 'No entries found in database. Calculation started!'
        eEnergies, weights = smr.EnergyWeightsMagnonsFalt(m, qRoh, qRohErw, Qg, q1g, q2g, q3g, t, DuD, Borient, NuclearBragg, QVector, Kvector).T
        smr.add_EW_to_table(BC2, t, int(qMax), Bfrac, Borient, NuclearBragg, QVector, Kvector, eEnergies, weights)
    return eEnergies/BC2, weights

###############################################################################
