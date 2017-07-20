#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jul 11 15:21:16 2017

@author: lbeddric
"""

""" What could possibly go wrong??? """

""" Another implementation of the original Mathematica Skyrmion Model by Markus Garst et. al """

"""

GENERAL COMMENTS:
    - is my set-up for the initial magnetization arrangement really correct??? --> CHECK!!

"""

###############################################################################
#######################                             ###########################
#######################                             ###########################
#######################    SKYRMION-MODEL 0.2.0     ###########################
#######################                             ###########################
#######################                             ###########################
###############################################################################


###############################################################################
#######################     Basic Imports           ###########################
###############################################################################

import numpy as np; np.set_printoptions(threshold = 50)
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from scipy.optimize import minimize
from scipy.linalg import orth

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

BC2 = 45.2919                                                                   # Bc2 as definded, without dipole interaction
Bx, By, Bz = 0., 0., BC2/2.                                                       # right now arbitrary values, in units of T
Bhom = np.array([Bx, By, Bz])
B = np.linalg.norm(Bhom)                                                        # external Bfield in e3 dir
dirNSky = Bhom/B

#------------------------------------------------------------------------------ # see page 6

nMax = 300
qMax = 2.1                                                                      # nMax=Anzahl moeglicher q-Vektoren, qMax=radius um Q=0 in dem alle betrachteten q-Vektoren liegen

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

###############################################################################

###############################################################################
#######################   basic helping routines    ###########################
###############################################################################

def radius(n,m):
    """
    Returns 2-norm of a vector n*Q1Start + m*Q2Start
    
    arguments:
                n(int/float):               prefactor of Q1Start
                m(int/float):               prefactor of Q2Start              
                    
    return:
                val(float):                 2-norm of the vector
    """
    return np.linalg.norm(n*Q1Start + m*Q2Start)

#------------------------------------------------------------------------------

def loadqInd(qMax, keine_zusaetzlichen_Ringe = True):
    """
    
    """
    path = os.path.join(index_path, "qRoh.ind")
#    path = os.getcwd() + u"\index_files\qRoh.ind"                                    # windows format
#    path = os.getcwd() + "i/ndex_files/qRoh.ind"                                    # linux format

    qRohErw = np.genfromtxt(path, dtype = np.int8, delimiter = ",")
    
    if keine_zusaetzlichen_Ringe == True:
        return np.asarray([i for i in qRohErw if radius(*i) < qMax])
    else:
        return loadqInd(qMax + keine_zusaetzlichen_Ringe, True)
        
#------------------------------------------------------------------------------

def q(i, qRoh, qRohErw, Q1, Q2):
    """
    Returns qStart vector of the ith entry for given lattice index set and basis
    
    arguments:
                i(int):                     entry of the qRohErw lattice index set
                qRoh(ndarray[mx2]):         lattice index set as produced by loadqInd
                qRohErw(ndarray[mx2]):      lattice index set as produced by loadqInd
                Q1(ndarray[3]):             basis vector 1
                Q2(ndarray[3]):             basis vector 2
                
    return:
                Q(ndarray[3]):              indices of preliminary lattice
    """
    if i == 0:
        return np.array([0.,0.,0.])                                             # was just "pass" before
    else:    
        if len(qRohErw) > len(qRoh):
            return qRohErw[i, 0] * Q1 + qRohErw[i, 1] * Q2
        else:
            return qRoh[i, 0] * Q1 + qRoh[i, 1] * Q2
        
#------------------------------------------------------------------------------
    
def mrStart1(k, i, idir = dirNSky):
    """
    Returns the vector components of the initial magnetization dependend on the external mag. field
    
    arguments:
                k(int):                     presumably a integer value (not understood so far by the author)
                i(int):                     presumably: i in [0,1,2] otherwise out of range
                idir(ndattay):              initial direction of the external B field
                    
    return:
                val(float):                 i-th component of the initial magnetization vector
    """
    if k == 0:
        return 0.1 * dirNSky[i]
    elif k >= 1 and k <= 3:
        return 0.1 * -dirNSky[i]
    else:
        return np.array([0.,0.,0.])[i]
    
#------------------------------------------------------------------------------

def mrStart(k, i = "all", idir = dirNSky):
    """
    assembles and returns the initial real part of the magnetization vector depending on external B field
    
    arguments:
                k(int):                     presumably a integer value (not understood so far by the author)
                i(int):                     index of the initial mr vector [0, 1, 2]
                idir(ndattay):              initial direction of the external B field
                    
    return:
                val(float):                 i-th component of the initial magnetization vector
    """
    if k == 0 and i == "all":
        return np.array([0.,0.,0.])
    elif k != 0 and i == "all":
        return np.array([mrStart1(k,0), mrStart1(k,1), mrStart1(k,2)])
    elif k == 0 and i == 2:
        return 0.1 * dirNSky[i]
    else:
        print "Error occured at mrStart! Unexpected configuration of k and i"

#------------------------------------------------------------------------------

def miStart1(k, i, idir):
    """
    
    """
    qInd = np.array([[0,0], [0,1], [0,-1], [1,0]], dtype = np.int8)
    Q1 = np.array([1.,0.,0.])
    Q2 = .5 * np.array([-1., np.sqrt(3), 0.])
    
    if k >= 1 and k <= 3:
        tempvec = np.cross(q(k, qInd, qInd, Q1, Q2), mrStart(k, "all", idir))
        tempn = np.linalg.norm(tempvec)
        return tempvec[i]/tempn * -0.1
    else:
        return np.array([0.,0.,0.])[i]
        
#------------------------------------------------------------------------------

def miStart(k, idir = dirNSky):
    """
    assembles and returns the initial imaginary part of the magnetization vector depending on external B field
    
    arguments:
                k(int):                     presumably a integer value (not understood so far by the author)
                i(int):                     presumably: i in [0,1,2] otherwise out of range
                idir(ndattay):              initial direction of the external B field
                    
    return:
                val(float):                 i-th component of the initial magnetization vector
    """
    return np.array([miStart1(k, 0, idir), miStart1(k, 1, idir), miStart1(k, 2, idir)])

#------------------------------------------------------------------------------

def initQ(q1, q2, q3, idir):
    """
    initializes 2D vector base (120°) depending on coordinate input q1,q2,q3
    DO NOT NORMALIZE THE LATTICE BASIS VECTORS!!
        
    arguments:
                q1(float):                  first entry of Q1
                q2(float):                  second entry of Q1
                q3(float):                  third entry of Q1
                idir(ndarray[3]):           initial direction of external magnetic field
                
    return:
                Q1(ndarray[3]):             basis vector 1 of the hex lattice
                Q2(ndarray[3]):             basis vector 2 of the hex lattice
    """
    ang = np.pi*2./3
    Q1 = np.array([q1,q2,q3])
    Q2 = rot_vec(Q1, idir, ang)
    
    return Q1, Q2

#------------------------------------------------------------------------------

def unique_entries(Q):
    """
    I OMITTED VALUES AFTER e-15 TO FULLFILL CONDITIONS... ALLOWED? BETTER TO NOT CALCULATE THE INDICES AGAIN SINCE THEY 
    SHOULD NOT CHANGE?
    returns indices of the unique entries in the lattice from which all other can be reconstructed by symmetry operations
        
    arguments:
                q1(float):                  first entry of Q1
                q2(float):                  second entry of Q1
                q3(float):                  third entry of Q1
                idir(ndarray[3]):           initial direction of external magnetic field
                qMax(float):                qMax=radius um Q=0 in dem alle betrachteten q-Vektoren liegen should be slightly larger than intended radius (eg: 5. --> 5.1)
                
    return:
                l(list):                    indices of the unique entries
    """
    l = []
    q_1 = Q[1]
    nq_1 = np.linalg.norm(q_1)
    for i in xrange(1, len(Q)):
        q_i = Q[i]
#        cond1 = np.vdot(q_i / np.linalg.norm(q_i), q_1 / nq_1) <= 1.
        cond1 = np.round(np.vdot(q_i / np.linalg.norm(q_i), q_1 / nq_1), 15) <= 1.
#        cond2 = np.vdot(q_i / np.linalg.norm(q_i), q_1 / nq_1) > np.cos(np.pi/3.)                    # cos(pi/3) = 0.5
        cond2 = np.round(np.vdot(q_i / np.linalg.norm(q_i), q_1 / nq_1), 15) > np.cos(np.pi/3.)
#        cond3 = np.vdot([0., 0., 1.], np.cross(q_i, q_1)) >= 0.
        cond3 = np.round(np.vdot([0., 0., 1.], np.cross(q_i, q_1)), 15) >= 0.
        if cond1 and cond2 and cond3:
            l.append(i)
    return l

#------------------------------------------------------------------------------

def buildmag0(uel, idir = dirNSky):
    """
    
    """
    mr = np.zeros((len(uel) +1 , 3)); mr[0,2] = mrStart(0, 2,idir)
    mi = np.zeros((len(uel) +1 , 3)); mi[0] = miStart(0, idir)
    for i in xrange(len(uel)):
        mi[i+1] = miStart(uel[i], idir)
        mr[i+1] = mrStart(uel[i], "all", idir)

    return mr + mi

#------------------------------------------------------------------------------

def rot_vec(vec, axdir, angle):
    """
    Rotates vector "vec" around an arbitrary direction "axdir" (going through origin) with an angle "angle"
    https://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    
    arguments:
                vec(ndarray[3]):            vector to be rotated
                axdir(ndarray[3]):          direction of rotating axis
                angle(float):               rotating angle in [0., 2*pi[
                    
    return:
                val(ndarray[3]):            rotated vector
    """
    u = axdir / np.linalg.norm(axdir)
    R = np.array([[np.cos(angle) + u[0]*u[0]*(1.-np.cos(angle)), u[0]*u[1]*(1.- np.cos(angle)) - u[2]*np.sin(angle), u[0]*u[2]*(1.-np.cos(angle) + u[1]*np.sin(angle))],
                  [u[1]*u[0]*(1.-np.cos(angle)) + u[2]*np.sin(angle), np.cos(angle) + u[1]*u[1]*(1.-np.cos(angle)), u[1]*u[2]*(1.-np.cos(angle)) - u[0]*np.sin(angle)],
                  [u[2]*u[0]*(1.-np.cos(angle)) - u[1]*np.sin(angle), u[2]*u[1]*(1.-np.cos(angle)) + u[0]*np.sin(angle), np.cos(angle) + u[2]*u[2]*(1.-np.cos(angle))]])
        
    return np.dot(R, vec)

#------------------------------------------------------------------------------

def initmarray(uel, mag0, Q):
    """
    Initializes the whole magnetization array from the initial magnetization given for the unique sector
    
    arguments:
                uel(list/ndarray):          indices of the unique entries in the hex lattice
                mag0(ndarray):              initial magnetization with mag0[i+1] corresponding to uniquel[i] || mag0[0] is the magnetization at Q = 0
                                            mag0 NNEEEEDDSS to be in complex values!
                Q(ndarray[m,3]):            hex-lattice vectors
                
    return:
                mag(ndarray[nQ, 3]):         entire magnetization array
    """
    
    nQ = len(Q)
    
    mag = np.empty((nQ, 3), dtype = np.complex)                          # initiatlizing mag-array and filling in first sector
    mag[0] = mag0[0]
    for i in xrange(len(uel)):
        mag[uel[i]] = mag0[i+1]
    
    for k in xrange(1, nQ):                                           # using symmetry to fill up magnetization array
        for l in uel:
            if np.allclose(Q[k], Q[l]):       
                mag[k] = deepcopy(mag[l])                       # symmetry 1: q[k] == Rot.q[l] --> m[k] = Rot.m[l]
                mag[k+1] = np.conjugate(deepcopy(mag[l]))       # symmetry 2: q[k] == -q[l] --> m[k] = conjugate(m[l])
            
            elif np.allclose(Q[k], rot_vec(Q[l], np.array([0., 0., 1.]), 2./3. * np.pi)):
                mag[k] = rot_vec(deepcopy(mag[l]), np.array([0., 0., 1.]), 2./3. * np.pi)
                mag[k-1] = rot_vec(np.conjugate(deepcopy(mag[l])), np.array([0., 0., 1.]), 2./3. * np.pi)
            
            elif  np.allclose(Q[k], rot_vec(Q[l], np.array([0., 0., 1.]), 4./3. * np.pi)):
                mag[k] = rot_vec(deepcopy(mag[l]), np.array([0., 0., 1.]), 4./3. * np.pi)
                mag[k+1] = rot_vec(np.conjugate(deepcopy(mag[l])), np.array([0., 0., 1.]), 4./3. * np.pi)
                
    return mag

#------------------------------------------------------------------------------

def magLoader(B, T):
    """
    Loading previously caculated magnetization for specified T (arb.u.) and B (arb.u.)
    """
    path = mag_path
#    path = os.getcwd() + u"\mag_database\\"                                        # windows format
#    path = os.getcwd() + "/mag_database/"                                          # linux format

    searchB = round(B, 3)
    
    d = np.genfromtxt(path + "B_%s,T_%s.out" %(str(searchB), str(T)), delimiter = ",")
    return d[0], d[1:]

#------------------------------------------------------------------------------

def indexMap(kvec, qRoh, qRohErw, Q1, Q2):
    """
    
    """
    IndexNewPosList = []
    minpos = np.argmin([np.linalg.norm(kvec-q(i, qRoh, qRohErw, Q1, Q2)) for i in xrange(len(qRohErw))])      # my solution not as in mathematica
    
    IndexPosList = np.asarray([np.where(np.all(qRohErw == qRohErw[i], axis = 1))[0] for i in xrange(len(qRohErw))], dtype = np.uint8)
    temp = [np.where(np.all(qRohErw + qRohErw[minpos] == qRohErw[j], axis = 1))[0] for j in xrange(len(qRohErw))]
    
    for k in temp:
        try:
            IndexNewPosList.append([k[0]])
        except IndexError:
            IndexNewPosList.append([None])
            
    return {"IndexPosList" : IndexPosList, "IndexNewPosList" : IndexNewPosList, "minpos" : minpos}

#------------------------------------------------------------------------------

###############################################################################

###############################################################################
#######################    Free energy routines     ########################### # endcontent routines
###############################################################################

def Phi4Term(qRoh, m):
    """
    
    """
    qRoh = np.array(qRoh, dtype = np.int8)
    nQ = len(qRoh) - 1
    S = 0.
    for i1 in xrange(nQ+1):#-1):
        for i2 in xrange(nQ+1):
            for i3 in xrange(nQ+1):
                for i4 in xrange(nQ+1):
                    if np.all(qRoh[i1] + qRoh[i2] + qRoh[i3] + qRoh[i4] == np.array([0, 0], dtype = np.int8)):
                        S += np.dot(m[i1], m[i2]) * np.dot(m[i3], m[i4])
    
    return S

#------------------------------------------------------------------------------

def Phi4Term2(m, Ringe = 2):
    """
    
    """
    path = os.path.join(index_path, "%i_Ringe_Phi4.ind" % Ringe)
#    path = os.getcwd() + u"\index_files\%i_Ringe_Phi4.ind" % Ringe                                        # windows format
#    path = os.getcwd() + "/index_files/%i_Ringe_Phi4.ind" % Ringe                                         # linux format    

    inds = np.genfromtxt(path, delimiter = ",", dtype = np.int8)
    
    S = 0.
    for i in inds:
        S += np.dot(m[i[0]], m[i[1]]) * np.dot(m[i[2]], m[i[3]])
    
    return S

#------------------------------------------------------------------------------

def Phi2Term(qRoh, m):
    """
    
    """
    qRoh = np.array(qRoh, dtype = np.int8)
    nQ = len(qRoh) - 1
    S = 0.
    for i in xrange(nQ+1):
        j = np.int(np.where(np.all(qRoh == -qRoh[i], axis = 1))[0])
        S += np.dot(m[i], m[j])
    return S
                
#------------------------------------------------------------------------------

def qPhi2Term(qRoh, m, q):
    """
    
    """
    qRoh = np.array(qRoh, dtype = np.int8)
    nQ = len(qRoh) - 1
    S = 0.
    for i in xrange(nQ+1):
        j = np.int(np.where(np.all(qRoh == -qRoh[i], axis = 1))[0])
        S += np.dot(q[i],q[i]) * np.dot(m[i], m[j])
    
    return S

#------------------------------------------------------------------------------

def DsyaTerm(qRoh, m, q):
    """
    
    """
    qRoh = np.array(qRoh, dtype = np.int8)
    nQ = len(qRoh) - 1
    S = 0.
    for i in xrange(nQ+1):
        k = np.int(np.where(np.all(qRoh == -qRoh[i], axis = 1))[0])
        S += -2.j * np.dot(m[i], np.cross(q[i], m[k]))
                
    return S

#------------------------------------------------------------------------------

def DipoleTerm(qRoh, m, q):
    """
    
    """
    qRoh = np.array(qRoh, dtype = np.int8)
    nQ = len(qRoh) - 1
    S = 0.
    for i in xrange(1, nQ+1):
        j = np.int(np.where(np.all(qRoh == -qRoh[i], axis = 1))[0])
        S += (np.dot(m[i], q[i]) * np.dot(m[j], q[i]))/np.dot(q[i], q[i])
    S += np.dot(m[0], np.dot(DemN, m[0]))
    
    return S
    
#------------------------------------------------------------------------------

def HighQcorr(qRoh, m, q, q1, q2, q3):
    """
    
    """
    qRoh = np.array(qRoh, dtype = np.int8)
    nQ = len(qRoh) - 1
    S = 0.
    for i in xrange(1, nQ+1):
        j = np.int(np.where(np.all(qRoh == -qRoh[i], axis = 1))[0])
        S += np.dot(np.dot(q[i],q[i]) * m[i],np.dot(q[j],q[j]) * m[j])/(q1**2 + q2**2 + q3**2)
                
    return S

#------------------------------------------------------------------------------

def free_energy(qRoh, m, q, q1, q2, q3, B, t, DuD):
    """
    Sum of all terms contributing to the total free energy.
    
    arguments:
                m(ndarray):                 magnetization at each (qRoh)-point of the hex lattice
                qRoh(list):                 indices of preliminary lattice
                q(ndarray[nQ+1,3]):         q-vectors of the extended preliminary lattice
                q1(float):                  first entry of Q1
                q2(float):                  second entry of Q1
                q3(float):                  third entry of Q1
                B(float):                   external magnetic field (as fraction of Bc2?)
                t(float):                   Look at Papers
                DuD(float):                 DM stiffness ... NEIN
                
    return:
                f(float):                   value of the free energy

    """

#    f = Phi4Term(qRoh, m)
    f = Phi4Term2(m, 2)                                                          # don't check all the indices again and again...
    f += (1.+t) * Phi2Term(qRoh, m)
    f += qPhi2Term(qRoh, m, q)
    f += DsyaTerm(qRoh, m, q)
    f += DuD/2. * DipoleTerm(qRoh, m, q)
    f -= 0.0073 * HighQcorr(qRoh, m, q, q1, q2, q3)
    f -= np.dot(B, m[0])
    return float(np.real(f))

###############################################################################

###############################################################################
#######################    minimization routines    ###########################
###############################################################################

def setupMinimization(x0, B, t, DuD, qRoh, qRohErw, idir, uel):
    """
    callable of the minimization routine
    mr[0,0] & mr[0,1] actually should not be part of optimization routine...
    """
    
#    q1, q2, q3 = x0[:3].astype(float)
    q1, q2 = x0[:2].astype(float)
    q3 = 0.                                                                     # add q3 being set to 0
    
#    mag0 = x0[3:].reshape((-1,3))
    magreal = np.concatenate(([0.,0.],x0[2:])).reshape((-1,3))                     # add mi[0,0] and mi[0,1] set to 0.
    mag0 = np.concatenate((magreal[:,:2] * 1.j, magreal[:,2].reshape((-1,1))), axis = 1)

    nQ = len(qRoh) - 1
    Q1, Q2 = initQ(q1, q2, q3, idir)
    Q = np.array([q(i, qRoh, qRohErw, Q1, Q2) for i in xrange(0,nQ+1)])
#    l = unique_entries(q1, q2, q3, idir, qMax)                                 # nicht jedes mal ne berechnen, macht das Mathematica programm auch nicht. hier nur fehlerquelle

    mag = initmarray(uel, mag0, Q)
    
    return free_energy(qRoh, mag, Q, q1, q2, q3, B, t, DuD)

#------------------------------------------------------------------------------

def groundState(q1, q2, q3, B, t, DuD, qRoh, qRohErw, mag0, idir, uel):
    """
    
    """
#    x0 = np.concatenate(([q1, q2, q3], mag0.flatten()))
    x0 = np.concatenate(([q1,q2], mag0.flatten()[2:]))                          # leave out q3 and mi[0,0], mi[0,1]
    
    erg = minimize(setupMinimization, x0, method = "BFGS", args = (B, t, DuD, qRoh, qRohErw, idir, uel), options = {"gtol" : 1e-7,"disp" : True})
    
    return erg

#------------------------------------------------------------------------------

def reswriter(xc, t, B):
    """
    
    """
    path = mag_path
#    path = os.getcwd() + u"\mag_database\\"                                        # windows format
#    path = os.getcwd() + "/mag_database/"                                          # linux format

#    headerstr = "the q's: q1 = %f, q2 = %f, q3 = %f \n mag[0] = mi[0], mag[1] = mi[1], mag[2] = mr[2] \n " %tuple(xc[:3])
    headerstr = "the q's: q1 = %f, q2 = %f, q3 = %f \n mag[:,0] = mi[0], mag[:,1] = mi[1], mag[2] = mr[2] B-Feld = %f \n " %(xc[0], xc[1], 0., B)
    mwrite = np.concatenate(([xc[0], xc[1], 0.],[0.,0.], xc[2:])).reshape((-1,3))
    np.savetxt(path + "B_%s,T_%s.out" %(str(round(B,3)), str(t)), mwrite, delimiter = ",", header = headerstr)
    
#------------------------------------------------------------------------------

###############################################################################

###############################################################################
#####################   VecBase/MatBaseTrafos(Sky)   ##########################
###############################################################################

def VecBaseTrafoSky(vec, B, kvec, qRoh, qRohErw, Q1, Q2):
    """
    How does it work in my lattice with its indices? :P
    
    arguments:
                vec(ndarray[3?]):           in the extended system
                B(?x?>1):                   ??? Brillouin ???
                kvec(ndarrray[3?]):         K-vector (also neg. possible)
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                
    return:     
                nv(?):                      ???
    """
    
    vecp = vec.reshape((-1, 3))     # casts vec into mx3
    
    if len(vecp) > len(qRohErw):
        print "Error: vec > Number of BZ in qRohErw"
        return None
    
    imap = indexMap(kvec, qRoh, qRohErw, Q1, Q2)
    IndexPosList = imap["IndexPosList"]
    IndexNewPosList = imap["IndexNewPosList"]
#    minpos = np.argmin([np.linalg.norm(kvec-q(i, qRoh, qRohErw, Q1, Q2)) for i in xrange(len(qRohErw))])      # my solution not as in mathematica
    kBZ = kvec - q(imap["minpos"], qRoh, qRohErw, Q1, Q2)
    
#    IndexPosList = np.asarray([np.where(np.all(qRohErw == qRohErw[i], axis = 1))[0] for i in xrange(len(qRohErw))], dtype = np.int8)
#    temp = [np.where(np.all(qRohErw - qRohErw[minpos] == qRohErw[j], axis = 1))[0] for j in xrange(len(qRohErw))]   # I changed "+" to "-" inside where(...)
#    IndexNewPosList = []
#    for k in temp:
#        try:
#            IndexNewPosList.append([k[0]])
#        except IndexError:
#            IndexNewPosList.append([None])
    
    nv =  np.zeros((len(qRohErw), 3), np.complex)
    for l in xrange(len(qRohErw)):
        if IndexNewPosList[l][0] != None and IndexPosList[l, 0] < len(vecp):
            nv[IndexNewPosList[l][0]] = vecp[IndexPosList[l, 0]]
            
    return nv

###############################################################################

###############################################################################
#####################       Fluctuation Matrix       ##########################
###############################################################################

def perm_parity(a,b):
    """
    helper function for Levi-Civita tensor
    
    copied from https://bitbucket.org/snippets/lunaticjudd/doqp7/python-implementation-of-levi-civita
    
    Modified from
    http://code.activestate.com/recipes/578236-alternatve-generation-of-the-parity-or-sign-of-a-p/
    """
    
    a = list(a)
    b = list(b)

    if sorted(a) != sorted(b): return 0
    inversions = 0
    while a:
        first = a.pop(0)
        inversions += b.index(first)
        b.remove(first)
    return -1 if inversions % 2 else 1

#------------------------------------------------------------------------------

def loop_recursive(dim,n,q,s,paritycheck):
    """
    helper function for recursive calculation of the Levi-Civita tensor
    
    copied from https://bitbucket.org/snippets/lunaticjudd/doqp7/python-implementation-of-levi-civita
    """
    if n < dim:
        for x in range(dim):
            q[n] = x
            loop_recursive(dim,n+1,q,s,paritycheck)
    else:
        s.append(perm_parity(q,paritycheck))
        
#------------------------------------------------------------------------------
        
def LeviCivitaTensor(dim):
    """
    Levi-Civita tensor for arbitrary dimensions
    
    arguments:
                dim(int):                   dimension of the levi-civita tensor
                
    return:     
                lct(ndarray[dxd]):          Levi-Civita tensor
    """
    qinit = np.zeros(dim)
    paritycheck = range(dim)
    flattened_tensor = []
    loop_recursive(dim,0,qinit,flattened_tensor,paritycheck)

    return np.reshape(flattened_tensor,[dim]*dim)

#------------------------------------------------------------------------------

def krondelta(i,j):
    """
    the mathematical kronecker delta
    """
    if i == j:
        return 1
    else:
        return 0

#------------------------------------------------------------------------------

def checkVecSum(qRoh, a1, n, nn):
    """
    seems to work fine
    helper function for calculating the fluctuation matrix returns indices required for the 2nd and 3rd term
    
    arguments:
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                a1(int):                    first index (neg)
                n(int):                     second index (pos)
                nn(int):                    third index (neg)
                
    return:     
                ind(int or None):           the index required for evaluation, or nothing --> no value added to sum
    """
    try:
        ind = np.int(np.where(np.all(qRoh == -qRoh[a1] - qRoh[nn] + qRoh[n], axis = 1))[0])
        return ind
    except TypeError:
        return None

#------------------------------------------------------------------------------

def g_ij(n, nn, i, j, kx, ky, kz, qRoh, mag, q, q1, q2, q3, t, DuD, B):
    """
    CURRENTLY NOT WORKING!
    calculates entries of the fluctuation matrix
    
    implement: give result of checkvecsum as argument to g_ij from fluctuationM
    
    for later optimization: check krondelta first --> calculate the terms only if needed!
    
    """
    mag = np.concatenate((mag[:,:2] * 1.j, mag[:,2].reshape((-1,1))), axis = 1)
    kBZ = np.array([kx, ky, kz])
    nQ = len(qRoh)
    
    gt11 = (1 + t + (np.dot(Q[n], Q[n]) + 2*np.dot(Q[n], kBZ) + np.dot(kBZ, kBZ)) \
            - 0.0073 * (np.dot(q[n],q[n]) + 2*np.dot(q[n], kBZ) + np.dot(kBZ, kBZ))**2/(q1**2 * q2**2)) \
            * krondelta(i, j) - 2.j * np.dot(LeviCivitaTensor(3)[i,j], q[n] + kBZ)
    if np.allclose(q[n] + kBZ, np.array([0., 0., 0.])):
        gt12 = DemN[i, j]
    else:
        gt12 = ((q[n] + kBZ)[i] * (q[n] + kBZ)[j])/np.dot(q[n] + kBZ, q[n] + kBZ)
        
    gt2 = 0.
    gt3 = 0.
    for a1 in xrange(nQ):
        ind = checkVecSum(qRoh, a1, n, nn)
        try:
            gt2 += np.dot(mag[a1], mag[ind])
            gt3 += mag[a1, i] * mag[ind, j]             # why should I run the loop twice? -> calc both at same time
        except IndexError:
            print "Not a valid index given for calculating fluctuation matrix"
            
    return krondelta(n, nn) * (gt11 + DuD/2. * gt12) + 2 * gt2 + 4 * gt3
    
#------------------------------------------------------------------------------

def fluctuationM(B, kvec, nQ, n, nn, i, j, kx, ky, kz, qRoh, mag, q, q1, q2, q3, t, DuD):
    """
    kvec limited to 1.BZ
    """
    M = [[[[g_ij(n, nn, i, j, kvec[0], kvec[1], kvec[2], qRoh, mag, q, q1, q2, q3, t, DuD, B) for i in xrange(3)] for j in xrange(3)] for n in xrange(nQ)] for nn in xrange(nQ)]
    M = np.array(M)
    return M

#------------------------------------------------------------------------------

def fluctuationMFalt():
    """
    
    """
    pass

###############################################################################
###############################################################################

###############################################################################
#####################   PLAYGROUND / PROGRAMM   ###############################
###############################################################################

np.set_printoptions(threshold = 1000)

qRoh = loadqInd(2.1); nQ = len(qRoh) - 1
qRohErw = loadqInd(2.1, 4.); nQErw = len(qRohErw) - 1

Q1, Q2 = initQ(q1,q2, q3, dirNSky)

Q = np.array([q(i, qRoh, qRohErw, Q1, Q2) for i in xrange(nQ+1)])

uel = unique_entries(Q)

mag0real = buildmag0(uel)                                                       # keine komplexen zahlen! weniger speicher und ansonsten keine kompatibilität mit MINIMIZE
#mag = initmarray2(uel, mag0, qRoh, qRohErw, Q1, Q2)

###############################################################################