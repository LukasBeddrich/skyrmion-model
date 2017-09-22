#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Tue Jul 11 15:21:16 2017

@author: lbeddric
"""

""" What could possibly go wrong??? """

""" Another implementation of the original Mathematica Skyrmion Model by Johannes Waizner and Markus Garst et. al """

"""

GENERAL COMMENTS:
- optimize computation time

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

import numpy as np; np.set_printoptions(threshold = 50, precision = 15)
import matplotlib.pyplot as plt
import os
import sqlite3
from copy import deepcopy
from scipy.optimize import minimize
from scipy.linalg import eigvals
from matplotlib import cm
import pathos.pools as pool
import multiprocessing
import ctypes
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
Bfrac = 0.5
Bx, By, Bz = 0., 0., BC2/2.                                                       # right now arbitrary values, in units of T
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

DuD = 2 * 0.34
t = -1000

#------------------------------------------------------------------------------

###############################################################################

###############################################################################
#######################   basic helping routines    ###########################
###############################################################################

def chop(a, precision = 1e-11):
    """
    
    """
    if a.dtype == np.complex:
        tempr = np.real(a)
        tempi = np.imag(a)
        
        return np.where(np.abs(tempr) < precision, np.zeros(a.shape), tempr) + np.where(np.abs(tempi) < precision, np.zeros(a.shape), tempi)*1.j
    elif a.dtype == np.float:
        return np.where(np.abs(a) < precision, np.zeros(a.shape), a)
    else:
        pass                                                                    # raise error?!

#------------------------------------------------------------------------------
#%%
def chop2(a, precision = 1e-11):
    """
    
    """
    a = list(a)
    for i in xrange(len(a)):
        try:
            a[i] = chop(a[i], precision)
        except AttributeError:
            print "not a numpy.ndarray! \n Probably not everything is chopped"
    return a
    """
    aa = []
    for i in a:
        try:
            aa.append(chop(i, precision))
        except AttributeError:
            print "not a numpy.ndarray! Probably not everything is chopped"
            
    return np.asarray(aa)
    """
#------------------------------------------------------------------------------
#%%
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
    initializing the magnetization for minimization
    
    arguments:
                uel(list):                  list of unique entries of  hex lattice points
                idir(ndarray[1x3]):         direction of magnetic field
                
    return:
                mag0real(ndarray[len(uel)+1xm]):
                                            initial magnetization of high symmetry points of the hex lattice but still REAL values
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
#%%
def skew_sym_mat(v):
    """
    gives the skew-symmetric cross-product matrix for v (in 3D)
    """
    tempm = np.zeros((3,3))
    tempm[0,1] = -v[2]
    tempm[1,0] = v[2]
    tempm[0,2] = v[1]
    tempm[2,0] = -v[1]
    tempm[1,2] = -v[0]
    tempm[2,1] = v[0]
    return tempm

#------------------------------------------------------------------------------
#%%
def find_rot_mat(vi, vf):
    """
    finds the rotation matrix to rotate vector vi in direction of vf
    """
    vin = vi/np.linalg.norm(vi)
    vfn = vf/np.linalg.norm(vf)    
    u = np.cross(vin, vfn)
    
    s = np.linalg.norm(u)
    c = np.dot(vin, vfn)
    ux = skew_sym_mat(u)
    
    if s != 0.:
        return np.eye(3) + ux + np.dot(ux, ux) * (1.-c)/(s*s)
    elif np.allclose(vin, vfn):
        return np.eye(3)
    elif np.allclose(vin, -vfn):
        return -np.eye(3)
    else:
        print "Error"
        return None

#------------------------------------------------------------------------------
#%%
def magtoimag(mag0real):
    """
    returns the complex magnetization for high symmetry lattice points
    
    arguments:
                mag0real(ndarray[len(uel)x3]):
                                            magnetization initialized by buildmag0
    return:
                mag0(ndarray[len(uel)x3]):  mag0real[:,:2] are nw imaginary
    """
    
    return np.concatenate((mag0real[:,:2] * 1.j, mag0real[:,2].reshape((-1,1))), axis = 1)
    
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

def magLoader(B, T, Ringe, mathematica = False):
    """
    Loading previously caculated magnetization for specified T (arb.u.) and B (arb.u.) and # of rings used
    
    arguments:
                B(float):                   magnetic field strength in unspecified units
                T(float):                   temperature in unspecified units
                Ringe(int):                 number of rings in reciprocal space, used for calculation
    
    return:
                q1, q2, q3(float):          components of the new Q1 lattice vector
                mag0realg(ndarray[mx3]):    real values of the magnetization of the high symmetry lattice points
    """
    path = mag_path
#    path = os.getcwd() + u"\mag_database\\"                                        # windows format
#    path = os.getcwd() + "/mag_database/"                                          # linux format

    searchB = round(B, 3)
    
    if mathematica:
        d = np.genfromtxt(os.path.join(path, "magmatica,Bfrac_%s,T_%s,R_%i.out" %(str(B), str(T), Ringe)))
    else:
        d = np.genfromtxt(os.path.join(path, "B_%s,T_%s,R_%i.out" %(str(searchB), str(T), Ringe)), delimiter = ",")
    return d[0], d[1:]

#------------------------------------------------------------------------------

def indexMap(kvec, qRoh, qRohErw, Q1, Q2):
    """
    relates the known properties of the hex lattice points with the new (shifted) ones
    
    arguments:
                kvec(ndarray[1x3]):         shift vector in the hex lattice
                qRoh(ndarray[mx2]):
                qRohErw(ndarray[nx2]):      
                Q1(ndarray[1x3]):           first basevector of the hex lattice
                Q2(ndarray[1x3]):           second basevector of the hex lattice
    
    return:
                dict(dic):                  "IndexPosList" : old indices of qRoh
                                            "IndexNewPosList" : new arangement of indices after shift
                                            "minpos" : qRoh index of nearest hex lattice point to kvec
                

    """
    IndexNewPosList = []
    minpos = np.argmin([np.linalg.norm(kvec-q(i, qRoh, qRohErw, Q1, Q2)) for i in xrange(len(qRohErw))])      # not entirely sure why a plus is needed here...
    
    IndexPosList = np.asarray([np.where(np.all(qRohErw == qRohErw[i], axis = 1))[0] for i in xrange(len(qRohErw))]) #NEEEEEEEEEEEEEEVER
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

def free_energy(qRoh, m, q, q1, q2, q3, B, t, DuD, Ringe):
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
    f = Phi4Term2(m, Ringe)                                                          # don't check all the indices again and again...
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

def setupMinimization(x0, B, t, DuD, qRoh, qRohErw, idir, uel, Ringe):
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
    
    return free_energy(qRoh, mag, Q, q1, q2, q3, B, t, DuD, Ringe)

#------------------------------------------------------------------------------

def groundState(q1, q2, q3, B, t, DuD, qRoh, qRohErw, mag0, idir, uel, Ringe):
    """
    calculates the magnetization of the groundstate. Every high symmetry point gets m
    """
#    x0 = np.concatenate(([q1, q2, q3], mag0.flatten()))
    x0 = np.concatenate(([q1,q2], mag0.flatten()[2:]))                          # leave out q3 and mi[0,0], mi[0,1]
    
    erg = minimize(setupMinimization, x0, method = "BFGS", args = (B, t, DuD, qRoh, qRohErw, idir, uel, Ringe), options = {"gtol" : 1e-7,"disp" : True})
    
    return erg

#------------------------------------------------------------------------------

def reswriter(xc, t, B, Ringe):
    """
    
    """
    path = mag_path
#    path = os.getcwd() + u"\mag_database\\"                                        # windows format
#    path = os.getcwd() + "/mag_database/"                                          # linux format

#    headerstr = "the q's: q1 = %f, q2 = %f, q3 = %f \n mag[0] = mi[0], mag[1] = mi[1], mag[2] = mr[2] \n " %tuple(xc[:3])
    headerstr = "the q's: q1 = %f, q2 = %f, q3 = %f \n mag[:,0] = mi[0], mag[:,1] = mi[1], mag[2] = mr[2] B-Feld = %f \n " %(xc[0], xc[1], 0., B)
    mwrite = np.concatenate(([xc[0], xc[1], 0.],[0.,0.], xc[2:])).reshape((-1,3))
    np.savetxt(os.path.join(path, "B_%s,T_%s,R_%i.out" %(str(round(B,3)), str(t), Ringe)), mwrite, delimiter = ",", header = headerstr)
    
#------------------------------------------------------------------------------

###############################################################################

###############################################################################
#####################   VecBase/MatBaseTrafos(Sky)   ##########################
###############################################################################

def VecBaseTrafoSky_new(vec, kvec, qRoh, qRohErw, Q1, Q2):
    """
    shifting/ transforming (mostly) row or colum vectors of the fluctuation matrix or Mx matrix such that the information 
    calculated from the normal system can be used for the whole extended system
    
    DOES kvec have any relation to an actual connection to reciprocal lattice and hence the scatteing of neutrons?
    
    arguments:
                vec(mod3(len(vec))=0):      vector to be transformed
                kvec(ndarrray[3?]):         K-vector (also neg. possible)
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                qRohErw(ndarray[nx2]):      
                Q1(ndarray[1x3]):           first basevector of the hex lattice
                Q2(ndarray[1x3]):           second basevector of the hex lattice
                
    return:     
                nv:                      transformed vector
    """
    
    vecp = vec.reshape((-1, 3))     # casts vec into mx3
    
    if len(vecp) > len(qRohErw):
        print "Error: vec > Number of BZ in qRohErw"
        return None
    
    imap = indexMap(kvec, qRoh, qRohErw, Q1, Q2)
    IndexPosList = imap["IndexPosList"]
    IndexNewPosList = imap["IndexNewPosList"]

#    kBZ = kvec - q(imap["minpos"], qRoh, qRohErw, Q1, Q2)
    
    nv =  np.zeros((len(qRohErw), 3), np.complex)
    #--------------------------------------------------------------------------
    def multi_order(INPL, sortfrom, sortin):
        IPL = range(len(qRohErw))
    for l in xrange(len(qRohErw)):
        if IndexNewPosList[l][0] != None and IndexPosList[l, 0] < len(vecp):    # first most important line
            nv[IndexNewPosList[l][0]] = vecp[IndexPosList[l, 0]]                # second most important line
            
    return nv

#------------------------------------------------------------------------------

def VecBaseTrafoSky(vec, kvec, qRoh, qRohErw, Q1, Q2):
    """
    shifting/ transforming (mostly) row or colum vectors of the fluctuation matrix or Mx matrix such that the information 
    calculated from the normal system can be used for the whole extended system
    
    DOES kvec have any relation to an actual connection to reciprocal lattice and hence the scatteing of neutrons?
    
    arguments:
                vec(mod3(len(vec))=0):      vector to be transformed
                kvec(ndarrray[3?]):         K-vector (also neg. possible)
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                qRohErw(ndarray[nx2]):      
                Q1(ndarray[1x3]):           first basevector of the hex lattice
                Q2(ndarray[1x3]):           second basevector of the hex lattice
                
    return:     
                nv:                      transformed vector
    """
    
    vecp = vec.reshape((-1, 3))     # casts vec into mx3
    
    if len(vecp) > len(qRohErw):
        print "Error: vec > Number of BZ in qRohErw"
        return None
    
    imap = indexMap(kvec, qRoh, qRohErw, Q1, Q2)
    IndexPosList = imap["IndexPosList"]
    IndexNewPosList = imap["IndexNewPosList"]

#    kBZ = kvec - q(imap["minpos"], qRoh, qRohErw, Q1, Q2)
    
    nv =  np.zeros((len(qRohErw), 3), np.complex)
    for l in xrange(len(qRohErw)):
        if IndexNewPosList[l][0] != None and IndexPosList[l, 0] < len(vecp):    # first most important line
            nv[IndexNewPosList[l][0]] = vecp[IndexPosList[l, 0]]                # second most important line
            
    return nv

#------------------------------------------------------------------------------

def MatBaseTrafo_new(mat, kvec, qRoh, qRohErw, Q1, Q2):
    """
    shifting/ transforming (mostly) the fluctuation matrix or Mx matrix such that the information 
    calculated from the normal system can be used for the whole extended system
    
    DOES kvec have any relation to an actual connection to reciprocal lattice and hence the scatteing of neutrons?
    
    arguments:
                vec(mod3(len(vec))=0):      vector to be transformed
                kvec(ndarrray[3?]):         K-vector (also neg. possible)
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                qRohErw(ndarray[nx2]):      
                Q1(ndarray[1x3]):           first basevector of the hex lattice
                Q2(ndarray[1x3]):           second basevector of the hex lattice
                
    return:     
                Tmat(ndarray[(nQ+1)*3x(nQ+1)*3]):
                                            transformed matrix
    """
    nQErwloc = len(qRohErw)
    mat_split1, mat_split2, mat_split3 = mat[:nQErwloc], mat[nQErwloc:2*nQErwloc], mat[2*nQErwloc:]
    id_start1, id_start2, id_start3 = 0, nQErwloc, 2*nQErwloc
    
    #--------------------------------------------------------------------------
    def parallel_Trafo(mat_split, id_start, real, imag):
        vl = len(mat_split[0])
        for i, vec in enumerate(mat):
            tempvec = VecBaseTrafoSky(vec, kvec, qRoh, qRohErw, Q1, Q2).flatten()
            real[(id_start + i)*vl:(id_start + i + 1)*vl] = np.real(tempvec).tolist()
            imag[(id_start + i)*vl:(id_start + i + 1)*vl] = np.imag(tempvec).tolist()
        return
    #--------------------------------------------------------------------------
    
    shared_tempm_real = multiprocessing.Array(ctypes.c_double, len(mat)*len(mat[0]))
    shared_tempm_imag = multiprocessing.Array(ctypes.c_double, len(mat)*len(mat[0]))
    
    
    p1 = multiprocessing.Process(target = parallel_Trafo, args = (mat_split1, id_start1, shared_tempm_real, shared_tempm_imag))
    p2 = multiprocessing.Process(target = parallel_Trafo, args = (mat_split2, id_start2, shared_tempm_real, shared_tempm_imag))
    p3 = multiprocessing.Process(target = parallel_Trafo, args = (mat_split3, id_start3, shared_tempm_real, shared_tempm_imag))
    
    p1.start()
    p2.start()
    p3.start()
    
    p1.join()
    p2.join()
    p3.join()
    
    
    return np.transpose(np.asarray(shared_tempm_real).reshape((len(mat)*len(mat[0]))) +\
                        np.asarray(shared_tempm_imag).reshape((len(mat)*len(mat[0]))) * 1.j)
    
#    tempm = np.transpose([VecBaseTrafoSky(i, kvec, qRoh, qRohErw, Q1, Q2).flatten() for i in mat])
    Tmat = np.transpose([VecBaseTrafoSky(j, kvec, qRoh, qRohErw, Q1, Q2).flatten() for j in tempm])
    return Tmat

#------------------------------------------------------------------------------

def MatBaseTrafo(mat, kvec, qRoh, qRohErw, Q1, Q2):
    """
    shifting/ transforming (mostly) the fluctuation matrix or Mx matrix such that the information 
    calculated from the normal system can be used for the whole extended system
    
    DOES kvec have any relation to an actual connection to reciprocal lattice and hence the scatteing of neutrons?
    
    arguments:
                vec(mod3(len(vec))=0):      vector to be transformed
                kvec(ndarrray[3?]):         K-vector (also neg. possible)
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                qRohErw(ndarray[nx2]):      
                Q1(ndarray[1x3]):           first basevector of the hex lattice
                Q2(ndarray[1x3]):           second basevector of the hex lattice
                
    return:     
                Tmat(ndarray[(nQ+1)*3x(nQ+1)*3]):
                                            transformed matrix
    """
    nQloc, nQErwloc = len(qRoh), len(qRohErw)
    tempm = np.array([VecBaseTrafoSky(i, kvec, qRoh, qRohErw, Q1, Q2).flatten() for i in mat[:3*len(qRoh)]])
    tempm = np.transpose(np.concatenate((tempm, np.zeros((3*(nQErwloc-nQloc),3*nQErwloc)))))
    Tmat = np.transpose([VecBaseTrafoSky(j, kvec, qRoh, qRohErw, Q1, Q2).flatten() for j in tempm])
    return Tmat

#------------------------------------------------------------------------------

def MatBaseTrafo_old(mat, kvec, qRoh, qRohErw, Q1, Q2):
    """
    shifting/ transforming (mostly) the fluctuation matrix or Mx matrix such that the information 
    calculated from the normal system can be used for the whole extended system
    
    DOES kvec have any relation to an actual connection to reciprocal lattice and hence the scatteing of neutrons?
    
    arguments:
                vec(mod3(len(vec))=0):      vector to be transformed
                kvec(ndarrray[3?]):         K-vector (also neg. possible)
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                qRohErw(ndarray[nx2]):      
                Q1(ndarray[1x3]):           first basevector of the hex lattice
                Q2(ndarray[1x3]):           second basevector of the hex lattice
                
    return:     
                Tmat(ndarray[(nQ+1)*3x(nQ+1)*3]):
                                            transformed matrix
    """
    tempm = np.transpose([VecBaseTrafoSky(i, kvec, qRoh, qRohErw, Q1, Q2).flatten() for i in mat])
    Tmat = np.transpose([VecBaseTrafoSky(j, kvec, qRoh, qRohErw, Q1, Q2).flatten() for j in tempm])
    return Tmat

#------------------------------------------------------------------------------

def MatBaseTrafo2(mat, kvec, qRoh, qRohErw, Q1, Q2):
    """
    shifting/ transforming (mostly) the fluctuation matrix or Mx matrix such that the information 
    calculated from the normal system can be used for the whole extended system
    
    DOES kvec have any relation to an actual connection to reciprocal lattice and hence the scatteing of neutrons?
    
    arguments:
                tempfullm(ndarray):         vector to be transformed
                kvec(ndarrray[3?]):         K-vector (also neg. possible)
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                qRohErw(ndarray[nx2]):      
                Q1(ndarray[1x3]):           first basevector of the hex lattice
                Q2(ndarray[1x3]):           second basevector of the hex lattice
                
    return:     
                Tmat(ndarray[(nQErw+1)*3x(nQErw+1)*3]):
                                            transformed matrix
    """
    nQdiff = len(qRohErw) - len(qRoh)
    tempfillm = np.asarray([np.concatenate((i, np.zeros(3*nQdiff))) for i in mat])
    tempfullm = np.concatenate((tempfillm, np.zeros((3*nQdiff,3*len(qRohErw)))))
    
    return MatBaseTrafo(tempfullm, kvec, qRoh, qRohErw, Q1, Q2)

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
        return .1

#------------------------------------------------------------------------------

def g_ij(n, nn, i, j, kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD, B):
    """
    USE g_ij2 INSTEAD!
    """
    mag = np.concatenate((mag[:,:2] * 1.j, mag[:,2].reshape((-1,1))), axis = 1)
    kBZ = np.array([kx, ky, kz])
    nQ = len(qRoh)
    
    gt11 = (1 + t + (np.dot(Q[n], Q[n]) + 2*np.dot(Q[n], kBZ) + np.dot(kBZ, kBZ)) \
            - 0.0073 * (np.dot(Q[n],Q[n]) + 2*np.dot(Q[n], kBZ) + np.dot(kBZ, kBZ))**2/(q1**2 + q2**2)) \
            * krondelta(i, j) - 2.j * np.dot(LeviCivitaTensor(3)[i,j], Q[n] + kBZ)
    if np.allclose(Q[n] + kBZ, np.array([0., 0., 0.])):
        gt12 = DemN[i, j]
    else:
        gt12 = ((Q[n] + kBZ)[i] * (Q[n] + kBZ)[j])/np.dot(Q[n] + kBZ, Q[n] + kBZ)
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

def g_ij2(n, nn, i, j,  kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD):
    """
    calculates the components of the fluctuation matrix.
    !!! The j index might be absolutely bad, because of j not being recognized as imaginary number "i" !!!
    !!! the ")" in gt11 right before newline and krondelta(i,j) closes, so the wrong thing is counted? !!!
    !!! check the whole damn calculation !!!
    
    arguments:
                n, nn(int):                 index in [0, len(qRoh)[
                i, j(int):                  index in [0,2[
                kx, ky, kz(float):          kompenents of q with Q = G + q where G is reciprocal lattice vector of hex lattice
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                mag(ndarray[mx3]):          mag[:,:2] need to be imaginary! magnetization derived from the result of groundState as given by initmarray(uel, magtoimag(mg0real), Q)
                Q(ndarray[mx3]):            hex lattice vectors in groundState
                q1, q2, q3(float):          components of Q1 after minimization process
                t(float):                   temperature (somehow)
                DuD(float):                 Dipole interaction strength
    """
    
    kBZ = np.array([kx, ky, kz])
    nQloc = len(qRoh)
    LCT = LeviCivitaTensor(3)
    
    gt11 = 0.
    gt12 = 0.
    gt2 = 0.
    gt3 = 0.
    
    if n == nn:
        gt11 = (1 + t + (np.dot(Q[n], Q[n]) + 2*np.dot(Q[n], kBZ) + np.dot(kBZ, kBZ)) \
            - 0.0073 * (np.dot(Q[n], Q[n]) + 2*np.dot(Q[n], kBZ) + np.dot(kBZ, kBZ))**2/(q1**2 + q2**2)) \
            * krondelta(i, j) - 2.j * np.dot(LCT[i,j], Q[n] + kBZ)
    
        if np.allclose(Q[n] + kBZ, np.array([0., 0., 0.])):
            gt12 = DemN[i, j]
        else:
            gt12 = ((Q[n] + kBZ)[i] * (Q[n] + kBZ)[j])/np.dot(Q[n] + kBZ, Q[n] + kBZ)
        
    for a1 in xrange(nQloc):
        ind = checkVecSum(qRoh, a1, n, nn)
        try:
            gt2 += np.dot(mag[a1], mag[ind])
            gt3 += mag[a1, i] * mag[ind, j]             # why should I run the loop twice? -> calc both at same time
        except IndexError:
            pass
            
    return gt11 + DuD/2. * gt12 + 2 * krondelta(i, j) * gt2 + 4 * gt3
    
    
#------------------------------------------------------------------------------

def g_ij3(n, nn, i, j,  kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD):
    """
    calculates the components of the fluctuation matrix.
    !!! The j index might be absolutely bad, because of j not being recognized as imaginary number "i" !!!
    !!! the ")" in gt11 right before newline and krondelta(i,j) closes, so the wrong thing is counted? !!!
    !!! check the whole damn calculation !!!
    
    arguments:
                n, nn(int):                 index in [0, len(qRoh)[
                i, j(int):                  index in [0,2[
                kx, ky, kz(float):          kompenents of q with Q = G + q where G is reciprocal lattice vector of hex lattice
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                mag(ndarray[mx3]):          mag[:,:2] need to be imaginary! magnetization derived from the result of groundState as given by initmarray(uel, magtoimag(mg0real), Q)
                Q(ndarray[mx3]):            hex lattice vectors in groundState
                q1, q2, q3(float):          components of Q1 after minimization process
                t(float):                   temperature (somehow)
                DuD(float):                 Dipole interaction strength
    """
    LCT = LeviCivitaTensor(3)
    kBZ = np.array([kx, ky, kz])
    nQloc = len(qRoh)
    
    gt11 = 0.
    gt12 = 0.
    gt2 = 0.
    gt3 = 0.
    
    if n == nn:
        gt11 = (1 + t + (np.dot(Q[n], Q[n]) + 2*np.dot(Q[n], kBZ) + np.dot(kBZ, kBZ)) \
            - 0.0073 * (np.dot(Q[n], Q[n]) + 2*np.dot(Q[n], kBZ) + np.dot(kBZ, kBZ))**2/(q1**2 + q2**2)) \
            * krondelta(i, j) - 2.j * np.dot(LCT[i,j], Q[n] + kBZ)
    
        if np.allclose(Q[n] + kBZ, np.array([0., 0., 0.])):
            gt12 = DemN[i, j]
        else:
            gt12 = ((Q[n] + kBZ)[i] * (Q[n] + kBZ)[j])/np.dot(Q[n] + kBZ, Q[n] + kBZ)
        
    for a1 in xrange(nQloc):
        ind = checkVecSum(qRoh, a1, n, nn)
        try:
            gt2 += np.dot(mag[a1], mag[ind])
            gt3 += mag[a1, i] * mag[ind, j]             # why should I run the loop twice? -> calc both at same time
        except IndexError:
            pass
            
    return gt11 + DuD/2. * gt12 + 2 * krondelta(i, j) * gt2 + 4 * gt3
    
    
#------------------------------------------------------------------------------
    
def fluctuationM(kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD):
    """
    kvec = np.array([kx, ky, lz]) limited to 1. BZ
    calculates the whole fluctuation Matrix (necessarily hermitian) with size 3nQx3nQ using multiple cores
    
    arguments:
                kx, ky, kz(float):          kompenents of q with Q = G + q where G is reciprocal lattice vector of hex lattice
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                mag(ndarray[mx3]):          mag[:,:2] need to be imaginary! magnetization derived from the result of groundState as given by initmarray(uel, magtoimag(mg0real), Q)
                Q(ndarray[mx3]):            hex lattice vectors in groundState
                q1, q2, q3(float):          components of Q1 after minimization process
                t(float):                   temperature (somehow)
                DuD(float):                 Dipole interaction strength
    
    return:
                fM(ndarray[3*len(qRoh)x3*len(qRoh)]):
                                            full, not shifted fluctuation matrix
                                            
    !!! INFO !!! I switched i and j to "j,i" in the call of g_ij2, better / "more accurate" would be to switch it in g_ij2 itself.
                 because of symmetry properties of the fM matrix it is probably equivalent!
    """
    nQloc = len(qRoh)
    fM = np.zeros((3*nQloc, 3*nQloc), dtype = np.complex)
    long_list = []
    
    #--------------------------------------------------------------------------
    def g_n_poolprep(n):
        nn = n
        subfMs = []
        while nn <= (nQloc-1):
            subfM = np.asarray([[g_ij2(n, nn, j, i, kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD) for i in (0,1,2)] for j in (0,1,2)], dtype = np.complex)
            subfMs.append(subfM)
            nn+=1
        return subfMs
    #--------------------------------------------------------------------------
    p = pool.ProcessPool(processes=3)
    long_list.append(p.map(g_n_poolprep, range(nQloc)))
    
    
    for idi, rows in enumerate(long_list[0]):
        for idj, el in enumerate(rows):
            fM[3*idi:3*(idi+1), 3*(idj+idi):3*(idj+idi+1)] = 2*el
    
    return np.triu(fM) + np.conjugate(np.transpose(np.triu(fM,1)))

#------------------------------------------------------------------------------

def fluctuationM_old(kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD):
    """
    kvec = np.array([kx, ky, lz]) limited to 1. BZ
    calculates the whole fluctuation Matrix (necessarily hermitian) with size 3nQx3nQ
    
    arguments:
                kx, ky, kz(float):          kompenents of q with Q = G + q where G is reciprocal lattice vector of hex lattice
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                mag(ndarray[mx3]):          mag[:,:2] need to be imaginary! magnetization derived from the result of groundState as given by initmarray(uel, magtoimag(mg0real), Q)
                Q(ndarray[mx3]):            hex lattice vectors in groundState
                q1, q2, q3(float):          components of Q1 after minimization process
                t(float):                   temperature (somehow)
                DuD(float):                 Dipole interaction strength
    
    return:
                fM(ndarray[3*len(qRoh)x3*len(qRoh)]):
                                            full, not shifted fluctuation matrix
                                            
    !!! INFO !!! I switched i and j to "j,i" in the call of g_ij2, better / "more accurate" would be to switch it in g_ij2 itself.
                 because of symmetry properties of the fM matrix it is probably equivalent!
    """
    
    nQloc = len(qRoh)
    
    fM = np.zeros((3*nQloc, 3*nQloc), dtype = np.complex)
    
    
    for n in xrange(nQloc):
        for nn in xrange(nQloc):
            subfM = np.asarray([[g_ij2(n, nn, j, i, kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD) for i in (0,1,2)] for j in (0,1,2)], dtype = np.complex)
            fM[3*n:3*n+3, 3*nn:3*nn+3] = deepcopy(subfM)
    

    return 2.*fM

#------------------------------------------------------------------------------

def fluctuationMFalt(kx, ky, kz, qRoh, qRohErw, mag, Q, q1, q2, q3, t, DuD):
    """
    kvec = np.array([kx, ky, lz]) not limited to 1. BZ any more
    calculates the whole fluctuation Matrix (necessarily hermitian) with size 3nQx3nQ
    
    arguments:
                kx, ky, kz(float):          kompenents of q with Q = G + q where G is reciprocal lattice vector of hex lattice
                qRoh(ndarray[mx2]):         lattice index set as produced by qIndex
                mag(ndarray[mx3]):          mag[:,:2] need to be imaginary! magnetization derived from the result of groundState as given by initmarray(uel, magtoimag(mg0real), Q)
                Q(ndarray[mx3]):            hex lattice vectors in groundState
                q1, q2, q3(float):          components of Q1 after minimization process
                t(float):                   temperature (somehow)
                DuD(float):                 Dipole interaction strength
    
    return:
                fM(ndarray[3*len(qRoh)x3*len(qRoh)]):
                                            full, possibly shifted fluctuation matrix
    """
    
    
    kvec = np.asarray([kx, ky, kz])
    minpos = indexMap(kvec, qRoh, qRohErw, Q[3], Q[1])["minpos"]                # Q1 == Q[3] and Q2 == Q[1] is in this convention always true
    
    kBZ = kvec - Q[minpos]
#    print 'kBZ = ', kBZ
    
    return MatBaseTrafo2(fluctuationM(kBZ[0], kBZ[1], kBZ[2], qRoh, mag, Q, q1, q2, q3, t, DuD), kvec, qRoh, qRohErw, Q[3], Q[1])

###############################################################################

###############################################################################
#####################     linear Algebra     ##################################
###############################################################################

def SelectedEigenvectors(mCross, maxcutoff = 0.995, retless = True):
    """
    Does what "SelectedEigenvectors", "MidSelectedEigenvectors", "UnSelectedEigenvectors" at once.
    Should do the trick for mCross as well as mCrossFalt
    
    arguments:
                mCross(ndarray):            mCross matrix of the magnetization
                maxcutoff(float):           selects the usefull eigenvectors, -values
                retless(bool):              True -> returns only usefull
    return:
                vecs(ndarray):              transposed array of orthonormalized eigenvectors first orthonormalized eigvec = vecs[:,0]
    """
    
    eigval, eigvec = np.linalg.eig(mCross)
    eigval = np.imag(eigval)
    eigvec = eigvec.T
    eigvalcut = max(eigval) * maxcutoff
    
    inds = np.argsort(eigval)
    eigval = eigval[inds]
    eigvec = eigvec[inds]
    
    eigvecmax, eigvecmid, eigvecun = [], [], []
    
    for i in xrange(len(eigval)/2 - 1):
        if np.abs(eigval[i]) < 0.01:
            eigvecun.append(eigvec[i])
            eigvecun.append(eigvec[-i-1])
        elif np.abs(eigval[i]) >= eigvalcut:
            eigvecmax.append(eigvec[i])
            eigvecmax.append(eigvec[-i-1])
        else:
            eigvecmid.append(eigvec[i])
            eigvecmid.append(eigvec[-i-1])
    
    """ So far working quite well! """
    """ Need to understand what mathematica orthogonalization does """
    
    if retless:
        return np.linalg.qr(np.asarray(eigvecmax).T)[0]
    else:
        return np.linalg.qr(np.asarray(eigvecmax).T)[0], np.linalg.qr(np.asarray(eigvecmid).T)[0], np.linalg.qr(np.asarray(eigvecun).T)[0]
    
    """ last 4 lines: summary of what i did in the console """

###############################################################################

###############################################################################
#####################       Mx Matrix       ###################################
###############################################################################

def positionAddQtoN(b, c, qRoh):
    """
    needed to include the additional momentum structure and the fouriercomponents in Mx
    
    arguments:
                b(int):                     one of the indices
                c(int):                     another index
                qRoh(ndarray[mx2]):         index pairs of the hex lattice
                
    return:
                ind(int):                   index which fullfills condition or -1
    """
    try:
        ind = np.int(np.where(np.all(qRoh == qRoh[b] + qRoh[c], axis = 1))[0])
        return ind
    except TypeError:
        return -1

#------------------------------------------------------------------------------

def mCrossMatrix(mag, qRoh):
    """
    calculates the crossMatrix one needs to reverse the loops, compared to mathematica code
    mathematica multi-loop statements work quite similar to listcomprehension!!!
    
    arguments:
                mag(ndarray[mx3]):          full, complex magnetization
                qRoh(ndarray[mx2]):         index pairs of the hex lattice
                
    return:
                mx(ndarray):                
    """
    
    nQloc = len(qRoh)
    LCT = LeviCivitaTensor(3)
    
    mx = np.zeros((3*(nQloc), 3*(nQloc)), dtype = np.complex)
    
    for l in xrange(3):
        for j in xrange(3):
            for i in xrange(3):
                for c in xrange(nQloc):
                    for b in xrange(nQloc):                                 
                        
                        pos = positionAddQtoN(b,c, qRoh)
                        if pos != -1:
                            mx[3*pos + i, 3*c + l] += LCT[i,j,l] * mag[b,j] # was (3)[i,j,l] before but due to difference in array building between mathematica and python...
                            # not working!!!
                        b+=1
    return mx

#------------------------------------------------------------------------------

def mCrossMatrix_old(mag, qRoh):
    """
    calculates the crossMatrix one needs to reverse the loops, compared to mathematica code
    mathematica multi-loop statements work quite similar to listcomprehension!!!
    
    arguments:
                mag(ndarray[mx3]):          full, complex magnetization
                qRoh(ndarray[mx2]):         index pairs of the hex lattice
                
    return:
                mx(ndarray):                
    """
    
    nQloc = len(qRoh)
    
    mx = np.zeros((3*(nQloc), 3*(nQloc)), dtype = np.complex)
    
    for l in xrange(3):
        for j in xrange(3):
            for i in xrange(3):
                for c in xrange(nQloc):                                         # nicht 100% sicher wegen nQloc
                    for b in xrange(nQloc):                                 
                        
                        pos = positionAddQtoN(b,c, qRoh)
                        if pos != -1:
                            mx[3*pos + i, 3*c + l] += LeviCivitaTensor(3)[i,j,l] * mag[b,j] # was (3)[i,j,l] before but due to difference in array building between mathematica and python...
                            # not working!!!
    return mx
                
#------------------------------------------------------------------------------

def mCrossMatrixFalt(mag, qRoh, qRohErw, kvec, Q1, Q2):
    """
    calculates the crossMatrix
    
    arguments:
                mag(ndarray[mx3]):          full, complex magnetization
                qRoh(ndarray[mx2]):         index pairs of the hex lattice
                qRohErw(ndarray[nx2]):      index pairs of the extended hex lattice
                kvec(ndarray[1x3]):         shift vector in the hex lattice
                Q1, Q2(ndarray[1x3]):       base vectors hex lattice in groundState
                
    return:
                mxFalt(ndarray):            Matrix with shifted origin
    """
    return MatBaseTrafo2(mCrossMatrix(mag, qRoh), kvec, qRoh, qRohErw, Q1, Q2)

###############################################################################

###############################################################################
#####################    inverse susceptibility    ############################
###############################################################################

def mCrossSel(mag, qRoh):
    """
    Hier weitermachen!
    """
    mx = mCrossMatrix(mag, qRoh)
    seleigvec = SelectedEigenvectors(mx)
    
    return chop(np.dot(np.dot(np.conjugate(np.transpose(seleigvec)), mx),seleigvec))

#------------------------------------------------------------------------------

def mCrossSelFalt(mag, qRoh, qRohErw, kvec, Q1, Q2):
    """
    
    arguments:
                Q1(ndarray[1x3]):           Q1 of the optimized system
                Q2(ndarray[1x3]):           Q2 of the optimized system
    """
    mxf = mCrossMatrixFalt(mag, qRoh, qRohErw, kvec, Q1, Q2)
    seleigvecf = SelectedEigenvectors(mxf)
    
    return chop(np.dot(np.dot(np.conjugate(np.transpose(seleigvecf)), mxf),seleigvecf))

#------------------------------------------------------------------------------

def chiInv0Sel(kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD):
    """
    
    """
    fM = fluctuationM(kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD)
    seleigvec = SelectedEigenvectors(mCrossMatrix(mag, qRoh))
    
    return chop(np.dot(np.dot(np.conjugate(np.transpose(seleigvec)), fM),seleigvec))
#------------------------------------------------------------------------------

def chiInv0SelFalt(kx, ky, kz, qRoh, qRohErw, mag, Q, q1, q2, q3, t, DuD):
    """
    
    """
    fMf = fluctuationMFalt(kx, ky, kz, qRoh, qRohErw, mag, Q, q1, q2, q3, t, DuD)
    seleigvec = SelectedEigenvectors(mCrossMatrixFalt(mag, qRoh, qRohErw, np.array([kx, ky, kz]), Q[3], Q[1]))
    
    return chop(np.dot(np.dot(np.conjugate(np.transpose(seleigvec)), fMf),seleigvec))

#------------------------------------------------------------------------------

def chiInvFullSel(eps, mag, qRoh, kx, ky, kz, Q, q1, q2, q3, t, DuD):
    """
    
    """
    return 1.j * eps * np.linalg.inv(mCrossSel(mag, qRoh)) + chiInv0Sel(kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD)

#------------------------------------------------------------------------------

def chiInvFullSelFalt(eps, mag, qRoh, qRohErw, kx, ky, kz, Q, q1, q2, q3, t, DuD):
    """
    
    """
    kvec = np.array([kx, ky, kz])
    return 1.j * eps * np.linalg.inv(mCrossSelFalt(mag, qRoh, qRohErw, kvec, Q[3], Q[1])) + chiInv0SelFalt(kx, ky, kz, qRoh, qRohErw, mag, Q, q1, q2, q3, t, DuD)

###############################################################################

###############################################################################
#####################     Energy Spectrum     #################################
###############################################################################

def energySpectrum(mag, qRoh, kx, ky, kz, Q, q1, q2, q3, t, DuD):
    """
    
    """
    temp = np.real(1.j * eigvals(np.dot(mCrossSel(mag, qRoh), chiInv0Sel(kx, ky, kz, qRoh, mag, Q, q1, q2, q3, t, DuD))))
    return np.sort(temp[np.where(temp > 0)[0]])

#------------------------------------------------------------------------------

def energySpectrumFalt(mag, qRoh, qRohErw, kx, ky, kz, Q, q1, q2, q3, t, DuD):
    """
    
    """
    temp = np.real(1.j * eigvals(np.dot(mCrossSelFalt(mag, qRoh, qRohErw, np.array([kx,ky,kz]), Q[3], Q[1]), chiInv0SelFalt(kx, ky, kz, qRoh, qRohErw, mag, Q, q1, q2, q3, t, DuD))))
    return np.sort(temp[np.where(temp > 0)[0]])

#------------------------------------------------------------------------------
#%%
def EnergyWeightsMagnons(mag, qRoh, Q, q1, q2, q3, t, DuD, Borient, NuclearBragg, QVector, Kvector):
    """
    EnergyWeightsMagnons(mag, qRoh, kx, ky, kz, Q, q1, q2, q3, t, DuD, B, Borient, NuclearBragg, QVector, Kvector)
    """
    print "Start"
    # The code uses a coordinate system where the magnetic field axis is along 001. RotB is the rotation matrix that rotates the magnetic field axis into 001
    RotB = find_rot_mat(Borient, np.asarray([0.,0.,1.]))
    
    # Additionally, since cubic anisotropies are neglected, the Q - structure points in an arbitrary direction. 
    # One now needs to rotate the system again to match a "real" Q - vector direction with the program - internal Q - direction, i.e. q[1]
    QInternal = Q[1]
    nQInternal = np.linalg.norm(Q[1])
    RotQ = find_rot_mat(np.dot(RotB, QVector), QInternal)
    
    # ogether this forms the rotation matrix, that translates Vectors of the "real world" into program internal vectors
    RotMat = np.dot(RotQ, RotB)
    
    # To make life easier, the entered Kvector is normalized to units of QVectors
    Kvec = Kvector * nQInternal
    KvecRotated = np.dot(RotMat, Kvec)
    
    # The nuclear Bragg vector in this coordinate system is then given by
    NuclearBraggRotated = np.dot(RotMat, NuclearBragg/np.linalg.norm(NuclearBragg))
    NuclearBraggRotated /= np.linalg.norm(NuclearBraggRotated)
    print "Lab system transformed into theory system"
    
    NM = np.zeros((3*len(qRoh), 3*len(qRoh)))
    
    # normalized nuclear Bragg vector
    NormGVector = np.concatenate((NuclearBraggRotated, np.zeros(3*(len(qRoh) - 1))))
    
    # calculate the energy spectrum at given kx,ky,kz
    espec = energySpectrum(mag, qRoh, KvecRotated[0], KvecRotated[1], KvecRotated[2], Q, q1, q2, q3, t, DuD)
    WeightEs = lambda i: np.linalg.eig(chiInvFullSel(espec[i], mag, qRoh, KvecRotated[0], KvecRotated[1], KvecRotated[2], Q, q1, q2, q3, t, DuD))
    
    # construct projection matrix that projects onto the first Brillouin zone and to the direction orthogonal to nuclear Bragg vector
    ProjMatrix = deepcopy(NM)
    ProjMatrix[:3,:3] = np.eye(3)
    ProjMatrix -= np.outer(NormGVector, NormGVector)
    
    # contruct list with [energy, weight]
    MxSel = mCrossSel(mag, qRoh)
    SelEV = SelectedEigenvectors(mCrossMatrix(mag, qRoh))
    EnergyWeight = []
    for i in xrange(len(espec)):
        WeightVal, WeightVec = chop2(WeightEs(i))
        inds = np.argsort(np.abs(WeightVal))[::-1]
        WeightVal, WeightVec = WeightVal[inds], WeightVec[:,inds]
        if WeightVal[-1] < 0.001:
           tempw = np.real(np.trace(np.dot(ProjMatrix, np.dot(np.dot(SelEV, np.dot(np.outer(WeightVec[:,-1], np.conjugate(WeightVec[:,-1])), 1.j * MxSel)), np.conjugate(np.transpose(SelEV))))))
           EnergyWeight.append([espec[i], tempw])
        else:
            print "Error!"
            break
    return np.array(EnergyWeight)

#------------------------------------------------------------------------------
    
def EnergyWeightsMagnonsFalt(mag, qRoh, qRohErw, Q, q1, q2, q3, t, DuD, Borient, NuclearBragg, QVector, Kvector):
    """
    EnergyWeightsMagnons(mag, qRoh, kx, ky, kz, Q, q1, q2, q3, t, DuD, B, Borient, NuclearBragg, QVector, Kvector)
    """
    print "Start"
    # The code uses a coordinate system where the magnetic field axis is along 001. RotB is the rotation matrix that rotates the magnetic field axis into 001
    RotB = find_rot_mat(Borient, np.asarray([0.,0.,1.]))
    
    # Additionally, since cubic anisotropies are neglected, the Q - structure points in an arbitrary direction. 
    # One now needs to rotate the system again to match a "real" Q - vector direction with the program - internal Q - direction, i.e. q[1]
    QInternal = Q[1]
    nQInternal = np.linalg.norm(Q[1])
    RotQ = find_rot_mat(np.dot(RotB, QVector), QInternal)
    
    # ogether this forms the rotation matrix, that translates Vectors of the "real world" into program internal vectors
    RotMat = np.dot(RotQ, RotB)
    
    # To make life easier, the entered Kvector is normalized to units of QVectors
    Kvec = Kvector * nQInternal
    KvecRotated = np.dot(RotMat, Kvec)
    
    # The nuclear Bragg vector in this coordinate system is then given by
    NuclearBraggRotated = np.dot(RotMat, NuclearBragg/np.linalg.norm(NuclearBragg))
    NuclearBraggRotated /= np.linalg.norm(NuclearBraggRotated)
    print "Lab system transformed into theory system"
    
    NM = np.zeros((3*len(qRohErw), 3*len(qRohErw)))
    
    # normalized nuclear Bragg vector
    NormGVector = np.concatenate((NuclearBraggRotated, np.zeros(3*(len(qRohErw) - 1))))
    
    # calculate the energy spectrum at given kx,ky,kz
    especf = energySpectrumFalt(mag, qRoh, qRohErw, KvecRotated[0], KvecRotated[1], KvecRotated[2], Q, q1, q2, q3, t, DuD)
    WeightEs = lambda i: np.linalg.eig(chiInvFullSelFalt(especf[i], mag, qRoh, qRohErw, KvecRotated[0], KvecRotated[1], KvecRotated[2], Q, q1, q2, q3, t, DuD))
    
    # construct projection matrix that projects onto the first Brillouin zone and to the direction orthogonal to nuclear Bragg vector
    ProjMatrix = deepcopy(NM)
    ProjMatrix[:3,:3] = np.eye(3)
    ProjMatrix -= np.outer(NormGVector, NormGVector)
    
    # contruct list with [energy, weight]
    MxSelf = mCrossSelFalt(mag, qRoh, qRohErw, KvecRotated, Q[3], Q[1])
    SelEVf = SelectedEigenvectors(mCrossMatrixFalt(mag, qRoh, qRohErw, KvecRotated, Q[3], Q[1]))
    EnergyWeight = []
    for i in xrange(len(especf)):
        WeightVal, WeightVec = chop2(WeightEs(i))
        inds = np.argsort(np.abs(WeightVal))[::-1]
        WeightVal, WeightVec = WeightVal[inds], WeightVec[:,inds]
        if WeightVal[-1] < 0.001:
           tempw = np.real(np.trace(np.dot(ProjMatrix, np.dot(np.dot(SelEVf, np.dot(np.outer(WeightVec[:,-1], np.conjugate(WeightVec[:,-1])), 1.j * MxSelf)), np.conjugate(np.transpose(SelEVf))))))
           EnergyWeight.append([especf[i], tempw])
        else:
            print "Error!"
            break
    return np.array(EnergyWeight)
#------------------------------------------------------------------------------
#%%
def write_EW(EW, ks, B, T, Kvector, QVector):
    pass

###############################################################################

###############################################################################
#####################     data base methods     ###############################
###############################################################################

def create_EW_connection():
    """
    
    """
    
    dbpath = os.path.join(package_path, 'EnergiesWeights.db')
    conn = sqlite3.connect(dbpath)
    return conn
    
#------------------------------------------------------------------------------
    
def create_EW_table(Ringe):
    """
    
    """
    tablename = 'R_%i' % Ringe
    conn = create_EW_connection()
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS ' + tablename + ' (BC2 REAL, Bfrac REAL, T REAL, Borient1 REAL, Borient2 REAL, Borient3 REAL, NuclearBragg1 REAL, NuclearBragg2 REAL, NuclearBragg3 REAL, QVector1 REAL, QVector2 REAL, QVector3 REAL, \
    Kvector1 REAL, Kvector2 REAL, Kvector3 REAL, Energy REAL, Weight REAL)')
    conn.commit()
    cur.close()
    conn.close()
    
#------------------------------------------------------------------------------
    
def vec_to_text(*args):
    pass
    
#------------------------------------------------------------------------------
def add_EW_to_table(BC2, T, Ringe, Bfrac, Borient, NuclearBragg, QVector, Kvector, energies, weights):
    """
    
    """
    tablename = 'R_%i' % Ringe
    conn = create_EW_connection()
    cur = conn.cursor()
    if len(energies) == len(weights):
        for i in xrange(len(energies)):
            try:
                cur.execute("INSERT INTO %s (BC2, Bfrac, T,\
                            Borient1, Borient2, Borient3, \
                            NuclearBragg1, NuclearBragg2, NuclearBragg3, \
                            QVector1, QVector2, QVector3, \
                            Kvector1, Kvector2, Kvector3, \
                            Energy, Weight) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)" % tablename, 
                            (np.round(BC2,3), Bfrac, T, \
                             Borient[0], Borient[1], Borient[2], \
                             NuclearBragg[0], NuclearBragg[1], NuclearBragg[2], \
                             QVector[0], QVector[1], QVector[2], \
                             Kvector[0], Kvector[1], Kvector[2], \
                             energies[i], weights[i]))
            except sqlite3.OperationalError:
                print 'Error during insertion'
                
    else:
        print "something went wrong. energies and weights need to have same number of values!"
    conn.commit()
    cur.close()
    conn.close()

#------------------------------------------------------------------------------
#%%
def select_EW_from_table(BC2, T, Ringe, Bfrac, Borient, NuclearBragg, QVector, relQV, Kvector, relKv):
    """
    
    """
    tablename = 'R_%i' % Ringe
    conn = create_EW_connection()
    cur = conn.cursor()
  
    def veclim(QVec, Kvec, relQ, relK):
        temp = np.concatenate((QVec*(1.-relQ), QVec*(1.+relQ), Kvec*(1.-relK), Kvec*(1.+relK)))
        for i in xrange(len(temp)):
            if temp[i] == 0 and (i + 1)%6 == 0:
                temp[i] = min(relQ, relK)
            elif temp[i] == 0 and (i+1)%3 == 0:
                temp[i] = -min(relQ, relK)
        return temp
    
    QV1min, QV2min, QV3min, QV1max, QV2max, QV3max, Kv1min, Kv2min, Kv3min, Kv1max, Kv2max, Kv3max = veclim(QVector, Kvector, relQV, relKv)
    
    try:
        print 'QVmin = ({}, {}, {}) \nQVmax = ({}, {}, {}) \nKvmin = ({}, {}, {}) \nKvmax = ({}, {}, {})'.format(\
                       *veclim(QVector, Kvector, relQV, relKv))
        cur.execute('SELECT Energy, Weight FROM %s WHERE BC2 == %f AND Bfrac == %f AND T == %f AND \
                    Borient1 == %f AND Borient2 == %f AND Borient3 == %f AND \
                    NuclearBragg1 == %f AND NuclearBragg2 == %f AND NuclearBragg3 == %f AND \
                    QVector1 BETWEEN %f AND %f AND QVector2 BETWEEN %f AND %f AND QVector3 BETWEEN %f AND %f AND\
                    Kvector1 BETWEEN %f AND %f AND Kvector2 BETWEEN %f AND %f AND Kvector3 BETWEEN %f AND %f'\
                    % (tablename, np.round(BC2, 3), Bfrac, T,\
                    Borient[0], Borient[1], Borient[2],\
                    NuclearBragg[0], NuclearBragg[1], NuclearBragg[2],\
                    QV1min, QV1max, QV2min, QV2max, QV3min, QV3max,\
                    Kv1min, Kv1max, Kv2min, Kv2max, Kv3min, Kv3max))
#        print 'marked data'
    except sqlite3.OperationalError:
        print 'Error during read out'
    except TypeError:
        print 'Error formating SQL Querry'
    finally:
        data = np.asarray(cur.fetchall())
#        print 'returned data such as: {}'.format(data[0,:])
        cur.close()
        conn.close()
    
    return data
#%%

###############################################################################

###############################################################################
#####################     visualizations     ##################################
###############################################################################


    
###############################################################################
###############################################################################

###############################################################################
#####################   PLAYGROUND / PROGRAMM   ###############################
###############################################################################
#%%
"""
np.set_printoptions(threshold = 1000)

qRoh = loadqInd(qMax); nQ = len(qRoh) - 1
qRohErw = loadqInd(qMax, 4.); nQErw = len(qRohErw) - 1

Q1, Q2 = initQ(q1,q2, q3, dirNSky)

Q = np.array([q(i, qRoh, qRohErw, Q1, Q2) for i in xrange(nQ+1)])

uel = unique_entries(Q)

mag0real = buildmag0(uel)                                                       # keine komplexen zahlen! weniger speicher und ansonsten keine kompatibilität mit MINIMIZE
#mag = initmarray2(uel, mag0, qRoh, qRohErw, Q1, Q2)


magmaticapath = os.path.join(mag_path, "magmatica_R_3.out")
q1g, q2g, q3g, = np.genfromtxt(magmaticapath, delimiter = ",")[0]
magmatica = np.genfromtxt(magmaticapath, delimiter = ",")[1:]
Q1g, Q2g = initQ(q1g, q2g, q3g, dirNSky)
Qg = np.array([q(i, qRoh, qRohErw, Q1g, Q2g) for i in xrange(nQ+1)])
m = initmarray(uel, magtoimag(magmatica), Qg)
"""
#------------------------------------------------------------------------------



###############################################################################
