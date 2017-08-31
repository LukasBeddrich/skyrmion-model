#!/usr/bin/env python2
# -*- coding: utf-8 -*-
#
# Sample Python S(q,w) module for (anti-)ferromagnetic dispersions
#
# @author Tobias Weber <tobias.weber@tum.de>
# @license GPLv2
# @date jun-2016
#

import math as ma

import numpy as np
import numpy.linalg as la
from numpy import array	# in global namespace so that Takin can access it

import scipy as sp
import scipy.constants as const
import Skyrmion


# -----------------------------------------------------------------------------
# dispersion
# -----------------------------------------------------------------------------

# kB in meV/K
kB = const.k / const.e * 1e3


# dispersion relations
def disp_ferro(q, D, offs):
	return D*q**2. + offs

def disp_antiferro(q, D, offs):
	return D*q + offs


# Gaussian peak
def gauss(x, x0, sig, amp):
	norm = (np.sqrt(2.*np.pi) * sig)
	return amp * np.exp(-0.5*((x-x0)/sig)**2.) / norm

# Bose factor
def bose(E, T):
	n = 1./(m.exp(abs(E)/(kB*T)) - 1.)
	if E >= 0.:
		n += 1.
	return n

# Bose factor which is cut off below Ecut
def bose_cutoff(E, T, Ecut=0.02):
	Ecut = abs(Ecut)

	if abs(E) < Ecut:
		b = bose(np.sign(E)*Ecut, T)
	else:
		b = bose(E, T)

	return b

# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Takin interface
# -----------------------------------------------------------------------------

# global variables which can be accessed / changed by Takin
g_Bo = np.array([0., 0., 1.])   # magnetic field orientation
#g_QSky = np.array([np.cos(np.pi/6.) ,np.sin(np.pi/6.) ,0.])  # Skyrmion Pinning direction
g_QSky = np.array([1., 1., 0.])
g_G = np.array([1., 1., 0.])	# Bragg peak

g_sig = 0.02		# linewidth
g_S0 = 1.			# intensity

g_inc_sig = 0.02	# incoherent width
g_inc_amp = 1.		# incoherent intensity

g_T = 300.			# temperature

g_bose_cut = 0.02	# cutoff energy for Bose factor

g_disp = 0			# which dispersion?

g_Escale_emp = 1./30    # empiric estimate to fit the energy scale!
#
# the init function is called after Takin has changed a global variable (optional)
#
def TakinInit():
	print("Init: G=" + repr(g_G) + ", T=" + repr(g_T))


#
# dispersion E(Q) and weight factor (optional)
#
def TakinDisp(h,k,l):
    """
    
    """
#    Borient = np.array([0.,0.,1.])
#    NuclearBragg = g_G
    try:
        Kvector = np.array([h, k, l])
        eEnergies, weights = np.asarray(Skyrmion.disp_skyrmion(g_Bo, g_G, g_QSky, Kvector))
        
    except ZeroDivisionError:
        return [0.,0.]
    
    return [[eEnergies, -eEnergies], [weights, weights]]    


#
# S(Q,E) function, called for every Monte-Carlo point
#
def TakinSqw(h, k, l, E):
    try:
#		print("h={0}, k={1}, l={2}, E={3}".format(h,k,l,E))
        [Ep_peak, Em_peak], [wp_peak, wm_peak] = TakinDisp(h,k,l)
        
        S_p = np.sum([gauss(E, Ep_peak[i], g_sig, g_S0*wp_peak[i]) for i in xrange(len(Ep_peak))])
        S_m = np.sum([gauss(E, Em_peak[i], g_sig, g_S0*wm_peak[i]) for i in xrange(len(Em_peak))])
            

#		S_p = gauss(E, Ep_peak, g_sig, g_S0*wp_peak)
#		S_m = gauss(E, Em_peak, g_sig, g_S0*wm_peak)
        incoh = gauss(E, 0., g_inc_sig, g_inc_amp)

        S = (S_p + S_m)*bose_cutoff(E, g_T, g_bose_cut) + incoh
#		print("S={0}".format(S))
        return S
    except ZeroDivisionError:
        return 0.

# -----------------------------------------------------------------------------


import os
print("Script working directory: " + os.getcwd())
print('')

# test
#print(TakinSqw(1.1, 0.9, 0., 0.4))