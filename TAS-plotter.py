# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 19:56:48 2017

@author: lukas
"""

###############################################################################
#######################                             ###########################
#######################                             ###########################
#######################     TAS-plotter  0.1.0      ###########################
#######################                             ###########################
#######################                             ###########################
###############################################################################


###############################################################################
#######################     Basic Imports           ###########################
###############################################################################

import numpy as np; np.set_printoptions(threshold = 50)
import matplotlib.pyplot as plt
import os

###############################################################################

###############################################################################
#######################     setting up pathes       ###########################
###############################################################################

"""
defining global pathes
1. directory of the script containing this function and hence the package!
2. directory of the index files
3. directory of the previously calculated magnetizations
"""
global package_path
global data_path
    
package_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(package_path, "index_files")

###############################################################################

class TAS_measurement(object):
    """
    
    """
    
    def __init__(self, dpath, fname):
        self.__dpath = dpath
        self.__fname = fname
        
    def __del__(self):
        print "Measurement %s is closed." % self.__fname
        
    def load_spec(self):
        self._specdata = np.genfromtxt(os.path.join(self.__dpath, self.__fname))
        
    def load_params(self):
        self._params = params
        
###############################################################################

class TAS_measurements_comb(TAS_measurement):
    """
    
    """

###############################################################################

class ParamConverter(object):
    """
    
    """
    
    self.sth = "sth"
















