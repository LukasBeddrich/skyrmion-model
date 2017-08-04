#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 09:40:31 2017

@author: lbeddric, tweber
"""

# ----------------------------------------------------------------------------
import numpy as np
import scipy as sp
import scipy.interpolate as interp


g = 2.
mu_b = 5.7883817577208e-02        # meV/T

TBC2 = np.concatenate((np.arange(-5001, -4000, 250),np.arange(-2001, -1200, 200)[:-1], np.arange(-1201, 0, 100), np.arange(-571, 0, 30)))
BC2 = np.array([100.891, 94.9, 95.6909, 93.1904, 90.1649, 63.785, 60.4999, 57.1907, 53.5718, 49.6707, 47.4909, 45.3709, 43.0909, 40.4938, 37.9909, 35.1809, 31.9968, 28.7509, 24.7308, 20.2729, 14.3168, 1.42985, 34.4829, 33.2919, 32.2929, 31.6308, 30.2999, 29.4909, 28.2909, 27.091, 26.0907, 24.7308, 23.4903, 22.2706, 20.7829, 19.1667, 17.5706, 15.6719, 13.5905, 11.1272, 7.93282, 1.42985])


def Hc2_T(T):
    T_B = np.array([0, 2, 4, 8, 12, 16, 20, 24, 26, 26.5, 27, 27.5, 28, 28.5, 29, 29.5, 30])
    HC2_B = np.array([0.60125, 0.60125, 0.600325, 0.594775, 0.58275, 0.5587, 0.531875, 0.485625, 0.4403, 0.42735, 0.411625, 0.39035, 0.363525, 0.337625, 0.319125, 0.2886, 0.259])
    fcn = interp.UnivariateSpline(T_B, HC2_B, k=5)
    return fcn(T)

def Escale(T):
        Hc2 = Hc2_T(T)
        return g * mu_b * Hc2


print(Escale(20.))
print(Escale(28.5))
# ----------------------------------------------------------------------------
