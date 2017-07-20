# -*- coding: utf-8 -*-

import skyrmion_model_routines as smr
import numpy as np

#------------------------------------------------------------------------------

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

qRoh = smr.loadqInd(2.1); nQ = len(qRoh) - 1
qRohErw = smr.loadqInd(2.1, 4.); nQErw = len(qRohErw) - 1

Q1, Q2 = smr.initQ(q1, q2, q3, dirNSky)

Q = np.array([smr.q(i, qRoh, qRohErw, Q1, Q2) for i in xrange(nQ+1)])

uel = smr.unique_entries(Q)

mag0real = smr.buildmag0(uel)                                                       # keine komplexen zahlen! weniger speicher und ansonsten keine kompatibilität mit MINIMIZE

#------------------------------------------------------------------------------

res = smr.groundState(q1, q2, q3, Bhom, t, DuD, qRoh, qRohErw, mag0real, dirNSky, uel)

print res.x