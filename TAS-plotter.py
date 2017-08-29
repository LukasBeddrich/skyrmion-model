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
import re

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
        try:
            self._specdata = np.genfromtxt(os.path.join(self.__dpath, self.__fname))
        except IOError:
            print 'The specified file was not found!'
        
###############################################################################

class MIRAmeasurement(TAS_measurement):
    """
    
    """
    def __init__(self, dpath, fname):
        """ """
        super(TAS_measurement, self).__init__(self, dpath, fname)
        
    def __del__(self):
        """ """
        super(TAS_measurement, self).__del__(self)
        
    def parse_metadata(self):
        """ """
        try:
            f = open(os.path.join(self.__dpath, self.__fname))
            metadict = {'timestamp' : str(f.readline()[-20:-1])}
        except IOError:
            print 'The specified file was not found!'
        except:
            print 'Unexpected error occured.'
            raise
        
        for line in f:
            l = TASMetadataStr(line)
                
    

###############################################################################

class TASMetadataStr(str):
    """
    
    """
    splitter = ':'
    keys = r'\s+[\S]+\s+:'
    status = r'\s[\S]+:'
    sing_num = r'[0-9]*\.[0-9]*'
    unit = r'\s[A-Z,a-z]+\n'
    def _split(self):
        return self.split(self.splitter)
    
    def conv_meta_dict_simp(self):
        if self[0] == '#':
            self._split()
        
    def basic_meta(self):
        if self[0] == '#':
            temp = self[1:-1].strip()._split()
            if temp[1:] != str:
                return {temp[0] : ':'.join(temp[1:])}
            else:
                print 'Shit!'
        



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
    title, cmd, par, dpar = open(fpath, "r").readlines()[13:17]
    
    ident = identifier(temp[0])
    d = temp[1:].astype(np.float)
    
    varinds = range(ident["PNT"] + 1, ident["M1"])
    
    if len(varinds) == 1:
        
        print title
        plt.figure(facecolor = "w")
        plt.title(cmd)
        plt.errorbar(d[:,varinds], d[:,ident["CNTS"]]/d[:,ident["M1"]], d[:,ident["CNTS"]]/d[:,ident["M1"]] / np.sqrt(d[:,ident["CNTS"]]), marker = "o", mfc = "r", mec = "k", ls = "None", ecolor = "k")
        plt.show()

    elif np.allclose(2.*d[varinds[0]], d[varinds[1]]):
        
        print title
        plt.figure(facecolor = "w")
        plt.title(cmd)
        plt.errorbar(d[:,varinds[1]], d[:,ident["CNTS"]]/d[:,ident["M1"]], d[:,ident["CNTS"]]/d[:,ident["M1"]] / np.sqrt(d[:,ident["CNTS"]]), marker = "o", mfc = "r", mec = "k", ls = "None", ecolor = "k")
        plt.show()
        
    elif bedingung:
        pass


###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



