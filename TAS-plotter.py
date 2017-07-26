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
global TASP_path
    
package_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(package_path, "index_files")
TASP_path = os.path.join(os.path.split(package_path)[0], "TASPdata")

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
"""
class ParamConverter(object):

    
    self.sth = "sth"
"""
###############################################################################

def identifier(lstr):
    """
    
    """
    ident = {}
    for i in xrange(len(lstr)):
        if lstr[i] not in ident:
            ident.update({lstr[i] : i})
    return ident

###############################################################################

def simpleplot(n):
    """
    
    """
    
    fpath = os.path.join(TASP_path, "tasp2017n003%i.dat" % n)    
    temp = np.genfromtxt(fpath, skip_header = 46, dtype = str)
    
    ident = identifier(temp[0])
    d = temp[1:].astype(np.float)
    
    varinds = range(ident["PNT"] + 1, ident["M1"])
    
    if len(varinds) == 1:
        
        plt.figure(facecolor = "w")
        plt.errorbar(d[:,varinds], d[:,ident["CNTS"]]/d[:,ident["M1"]], d[:,ident["CNTS"]]/d[:,ident["M1"]] / np.sqrt(d[:,ident["CNTS"]]), marker = "o", mfc = "r", mec = "k", ls = "None", ecolor = "k")
        plt.show()













