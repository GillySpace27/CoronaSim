# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:13:05 2016


@author: chgi7364
"""

#print('Loading Dependencies...')
import numpy as np
import os
import sys
import copy

import matplotlib as mpl
#mpl.use('qt4agg')
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#import chianti.core as ch

from scipy import io
from scipy import ndimage
from scipy import interpolate as interp
from scipy.stats import norm
import scipy.stats as stats
from scipy.optimize import curve_fit
import copy

#import warnings
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    import astropy.convolution as con


from collections import defaultdict


import gridgen as grid
import progressBar as pb
import math
import time
import pickle
import glob

#from astropy import units as u
# import skimage as ski
# from skimage.feature import peak_local_max

import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
import platform
from mpi4py import MPI

np.seterr(invalid = 'ignore')

def absPath(path):
    #Converts a relative path to an absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))   
    return os.path.join(script_dir, path)    

####################################################################
##                          Environment                           ##
####################################################################

#Environment Class contains simulation parameters
class environment:
    #Environment Class contains simulation parameters

    #Locations of files to be used
    slash = os.path.sep
    datFolder = os.path.abspath("../dat/data/")

    def_Bfile = os.path.join(datFolder, 'mgram_iseed0033.sav')
    def_xiFile1 = os.path.join(datFolder, 'new_xi1.dat')
    def_xiFile2 = os.path.join(datFolder, 'new_xi2.dat')
    def_bkFile = os.path.join(datFolder, 'gilly_background_cvb07.dat')
    def_ioneq = os.path.join(datFolder, 'formattedIoneq.tsv')
    def_abund = os.path.join(datFolder, 'abundance.tsv')
    def_f1File = os.path.join(datFolder, 'f1_fix.txt')
    def_f2File = os.path.join(datFolder, 'f2_fix.txt')
    def_ionpath = os.path.abspath('../chianti/chiantiData/')
    def_hahnFile = os.path.join(datFolder, 'hahnData.txt')

    #For doing high level statistics
    fullMin = 0
    fullMax = 0
    fullMean = 0
    fullMedian = 0
    mapCount = 0

    #For randomizing wave angles/init-times
    primeSeed = 27
    randOffset = 0  

    timeRand = np.random.RandomState(primeSeed*2)
    streamRand = np.random.RandomState() #Gets seeded by streamindex
    primeRand = np.random.RandomState(primeSeed)
    

    #Constants
    c = 2.998e10 #cm/second (base velocity unit is cm/s)
    hev = 4.135667662e-15 #eV*s
    KB = 1.380e-16 #ergs/Kelvin
    r_Mm = 695.5 #Rsun in Mega meters
    mH = 1.67372e-24 #grams per hydrogen
    mE = 9.10938e-28 #grams per electron
    mP = 1.6726218e-24 #grams per proton

    #Parameters
    rstar = 1
    B_thresh = 6.0
    fmax = 8.2
    theta0 = 28
    S0 = 7.0e5

    #Element Being Observed
    mI = 9.2732796e-23 #grams per Iron
    ionString = 'fe'
    element = 26
    ion = 11
    lower = 1 #lower energy level
    upper = 38 #upper energy level

    #LamAxis Stuff   #######################
    Ln = 600
    lam0 = 188.217
    lamPm = 0.5

    psfSig = 0.047 #Angstroms


    def __init__(self, Bfile = None, bkFile = None, analyze = False, name = "Default", fFile = None):
        #Initializes
        self.name = name
        self._bfileLoad(Bfile, plot=False)
        self.__processBMap(thresh = 0.9, sigSmooth = 4, plot = False, addThresh = False)
        if analyze: self.analyze_BMap2()
        self._xiLoad()
        self._plasmaLoad(bkFile)
        self._chiantiLoad()
        self._fLoad(fFile)
        self._hahnLoad()
        self.makeLamAxis(self.Ln, self.lam0, self.lamPm)
        self.rtPi = np.sqrt(np.pi)

        print("Done")
        print('')

  ## File IO ##########################################################################

    def _bfileLoad(self, Bfile, plot = False):
        #Load Bmap
        if Bfile is None: self.Bfile = self.def_Bfile 
        else: self.Bfile = self.__absPath(Bfile)
        self.thisLabel = self.Bfile.rsplit(os.path.sep, 1)[-1] 

        print('Processing Environment: ' + str(self.thisLabel) +'...', end = '', flush = True)

        Bobj = io.readsav(self.Bfile)
        self.BMap_x = Bobj.get('x_cap')
        self.BMap_y = Bobj.get('y_cap')
        self.BMap_raw = Bobj.get('data_cap')

        if plot:
            plt.imshow((np.abs(self.BMap_raw)))
            plt.colorbar()
            plt.show()

    def _plasmaLoad(self, bkFile = None):
        #Load Plasma Background
        if bkFile is None: self.bkFile = self.def_bkFile 
        else: self.bkFile = self.__absPath(bkFile)
        x = np.loadtxt(self.bkFile, skiprows=10)
        self.bk_dat = x
        self.rx_raw = x[:,0]
        self.rho_raw = x[:,1]
        self.ur_raw = x[:,2]
        self.vAlf_raw = x[:,3]
        self.T_raw = x[:,4]

    def _fLoad(self, fFile = None):
        if fFile is None: 
            f1File = self.def_f1File
            f2File = self.def_f2File
            f3File = self.def_f3File
        else: 
            f1File = os.path.join(self.datFolder, 'f1_{}.txt'.format(fFile))
            f2File = os.path.join(self.datFolder, 'f2_{}.txt'.format(fFile))
            f3File = os.path.join(self.datFolder, 'f3_{}.txt'.format(fFile))
        x = np.loadtxt(f1File)
        y = np.loadtxt(f2File)
        z = np.loadtxt(f3File)
        self.fr = x[:,0]
        self.f1_raw = x[:,1]
        self.f2_raw = y[:,1]
        self.f3_raw = z[:,1]
        #print(f1File)
        #plt.plot(self.fr, self.f1_raw, label = "f1")
        #plt.plot(self.fr, self.f2_raw, label = "f2")
        #plt.axhline(1, color = 'k')
        #plt.legend()
        #plt.show()

    def _xiLoad(self):
        #Load Xi
        self.xiFile1 = self.def_xiFile1 
        self.xiFile2 = self.def_xiFile2 
        #plt.figure()

        x = np.loadtxt(self.xiFile1, skiprows=1)
        self.xi1_t = x[:,0]
        self.xi1_raw = x[:,1]
        self.last_xi1_t = x[-1,0]
        #plt.plot(self.xi1_t, self.xi1_raw)

        y = np.loadtxt(self.xiFile2, skiprows=1)
        self.xi2_t = y[:,0]
        self.xi2_raw = y[:,1]
        self.last_xi2_t = y[-1,0]

        self.tmax = int(min(self.last_xi1_t, self.last_xi2_t))
        #plt.plot(self.xi2_t, self.xi2_raw)
        #plt.show()
        pass

    def _chiantiLoad(self):
        ##Load Chianti File Info

        #Load in the ionization fraction info
        chi = np.loadtxt(self.def_ioneq)

            #Find the correct entry
        for idx in np.arange(len(chi[:,0])):
            if chi[idx,0] == self.element and chi[idx,1] == self.ion: break
        else: raise ValueError('Ion Not Found in fraction file')

        self.chTemps = chi[0,2:]
        self.chFracs = chi[idx,2:] + 1e-100

                    #Spline it!
        #self.splinedChTemps = np.linspace(min(self.chTemps),max(self.chTemps), 1000)
        #self.splinedChFracs = interp.spline(self.chTemps, self.chFracs, self.splinedChTemps)

        #T = []
        #for tt in self.chTemps:
        #    T.append(self.interp_frac(10**tt))
        #T = np.asarray(T)
        #print(self.chTemps)

        #plt.plot(self.chTemps, self.chFracs, 'o')
        #plt.title("Ion Fraction f(T) for " + str(self.ionString) + str(self.ion))
        #plt.xlabel("Temperature")
        #plt.ylabel("Fraction")
        #plt.yscale('log')
        ##plt.xscale('log')
        #plt.plot(self.chTemps,T)
        #plt.show()

        #Load in elemental abundance info
        abund = np.loadtxt(self.def_abund, usecols=[1])
        self.abundance = 10**(abund[self.element-1]-abund[0])

        #Load in upsilon(T) info
        fullstring = self.ionString + '_' + str(self.ion)
        ionpath = (self.def_ionpath +'/'+ self.ionString + '/' + fullstring)
        fullpath = os.path.normpath(ionpath + '/'+ fullstring + '.scups')

        getTemps = False
        getUps = False
        with open(fullpath) as f:
            for line in f:
                data = line.split()
                if getUps == True:
                    self.ups = [float(x) for x in data]
                    break

                if getTemps == True:
                    self.upsTemps = [float(x) for x in data]
                    getUps = True
                    continue

                if data[0] == str(self.lower) and data[1] == str(self.upper):
                    self.upsInfo = [float(x) for x in data]
                    getTemps = True

        self.splinedUpsX = np.linspace(0,1,200)
        self.splinedUps = interp.spline(self.upsTemps, self.ups, self.splinedUpsX)

        #Load in statistical weights
        fullpath2 = os.path.normpath(ionpath + '/'+ fullstring + '.fblvl')   
        with open(fullpath2) as f2:     
            for line in f2:
                data = line.split()
                if data[0] == str(self.lower):
                    self.wi = float(data[6])
        #print('')
        #print(self.ryd2ang(info[2]))
        #print(temps)
        #print(ups)  
        pass     

    def _hahnLoad(self):
        x = np.loadtxt(self.def_hahnFile)
        self.hahnAbs = x[:,0]
        line1 = x[:,1]
        line2 = x[:,2]

        self.hahnPoints = line1
        self.hahnError = line2-line1

  ## Magnets ##########################################################################
        
    def __processBMap(self, thresh = 0.9, sigSmooth = 4, plot = False, addThresh = False):

        #Gaussian smooth the image
        if sigSmooth == 0:
            self.BMap_smoothed = self.BMap_raw
        else:
            self.BMap_smoothed = ndimage.filters.gaussian_filter(self.BMap_raw, sigSmooth)

        #Find all above the threshold and label
        bdata = np.abs(self.BMap_smoothed)
        blist = bdata.flatten().tolist()
        bmean =  np.mean([v for v in blist if v!=0])
        bmask = bdata > bmean * thresh
        label_im, nb_labels = ndimage.label(bmask)
        
        #Create seeds for voronoi
        coord = ndimage.maximum_position(bdata, label_im, np.arange(1, nb_labels))

        #Get voronoi transform
        self.label_im, self.nb_labels, self.voroBMap = self.__voronoify_sklearn(label_im, coord, bdata)
        
        if addThresh:
            #Add in threshold regions
            highLabelIm = label_im + self.nb_labels
            self.label_im *= np.logical_not(bmask)
            self.label_im += highLabelIm * bmask
        
        #Clean Edges
        validMask = self.BMap_raw != 0
        self.label_im *= validMask
        self.voroBMap *= validMask
        self.BMap = self.voroBMap  

        if plot: #Plot Slice of Map
            #(label_im + 40) * validMask 
            plt.imshow(self.voroBMap, cmap = 'jet', interpolation='none')
            plt.colorbar()
            plt.title('Voronoi Average')
            #plt.imshow(highLabelIm, cmap = 'Set1', interpolation='none', alpha = 0.5)
            #plt.xlim(600,950)
            #plt.ylim(350,650)
            #coordinates = []
            #for co in coord: coordinates.append(co[::-1])
            #for co in coordinates:
            #    plt.scatter(*co, c='r')
            plt.show()
            #plt.figure()
            #plt.imshow(self.BMap_smoothed)
            #plt.colorbar()
            #plt.show()
            pass
        return

    def __voronoify_sklearn(self, I, seeds, data):
        import sklearn.neighbors as nb
        #Uses the voronoi algorithm to assign stream labels
        tree_sklearn = nb.KDTree(seeds)
        pixels = ([(r,c) for r in range(I.shape[0]) for c in range(I.shape[1])])
        d, pos = tree_sklearn.query(pixels)
        cells = defaultdict(list)

        for i in range(len(pos)):
            cells[pos[i][0]].append(pixels[i])

        I2 = I.copy()
        I3 = I.copy().astype('float32')
        label = 0
        for idx in cells.values():
            idx = np.array(idx)
            label += 1
            mean_col = data[idx[:,0], idx[:,1]].mean() #The pretty pictures part
            I2[idx[:,0], idx[:,1]] = label
            I3[idx[:,0], idx[:,1]] = mean_col
        #for point in I3: print(point)
        return I2, label, I3

    def analyze_BMap(self):
        #print('')
        #Find the number of pixels for each label
        labels = np.arange(0, self.nb_labels+1) - 0.5
        hist, bin_edges = np.histogram(self.label_im, bins = labels)
        #Get rid of region zero
        hist = np.delete(hist, 0)
        labels = np.delete(labels, [0, self.nb_labels])

        ##Plot Hist
        #plt.bar(labels, hist)
        #plt.xlabel('Region Label')
        #plt.ylabel('Number of Pixels')
        #plt.show()

        #Find a histogram of the region areas in terms of pixel count
        bins2 = np.arange(0, np.max(hist))
        hist2, bin_edges2 = np.histogram(hist, bins = 30)

        area_pixels = np.delete(bin_edges2, len(bin_edges2)-1)

        ##Plot Hist
        #width = (area_pixels[1] - area_pixels[0])*0.8
        #plt.bar(area_pixels, hist2 , width = width)
        #plt.xlabel('Pixel Area')
        #plt.ylabel('Number of Regions')
        #plt.show()

        #Find the area of a pixel in Mm
        pixWidth_rx = np.abs(self.BMap_x[1] - self.BMap_x[0])
        pixWidth_Mm = self.r_Mm * pixWidth_rx
        pixArea_Mm = pixWidth_Mm**2

        #Convert the area in pixels to an equivalent radius in Mm
        area_Mm = area_pixels * pixArea_Mm
        radius_Mm = np.sqrt(area_Mm/ np.pi)

        #Plot Hist

        plt.plot(radius_Mm, hist2, label = self.thisLabel)
        plt.title('Distribution of Region Sizes')
        plt.xlabel('Radius (Mm)')
        plt.ylabel('Number of Regions')
        plt.legend()
        #plt.show()

    def analyze_BMap2(self, NENV = 6):
        fullMap = np.abs(self.BMap_smoothed.flatten())
        thisMap = [x for x in fullMap if not x == 0]

        min = np.min(thisMap)
        max = np.max(thisMap)
        mean = np.mean(thisMap)
        median = np.median(thisMap)


        bmin = 2
        bmax = 50

        inside = 100*len([x for x in thisMap if bmin<x<bmax])/len(thisMap)

        environment.fullMin += min
        environment.fullMax += max
        environment.fullMean += mean
        environment.fullMedian += median
        environment.mapCount += 1

        plt.hist(thisMap, histtype = 'step', bins = 100, label = "%s, %0.2f%%" % (self.thisLabel, inside))

        if environment.mapCount == NENV:
            environment.fullMin /= NENV
            environment.fullMax /= NENV
            environment.fullMean /= NENV
            environment.fullMedian /= NENV

            plt.axvline(bmin)
            plt.axvline(bmax)
            plt.yscale('log')
            plt.xlabel('Field Strength (G)')
            plt.ylabel('Number of Pixels')
            plt.suptitle('Histograms of the Field Strengths')
            plt.title('Mean: ' + str(environment.fullMean) + ', Median: ' + str(environment.fullMedian) +
                        '\nMin: ' + str(environment.fullMin) + ', Max: ' + str(environment.fullMax))
            plt.legend()
            plt.show()

  ## Light ############################################################################

    def makeLamAxis(self, Ln = 100, lam0 = 200, lamPm = 0.5):
        self.lam0 = lam0
        self.lamAx = np.linspace(lam0 - lamPm, lam0 + lamPm, Ln)
        return self.lamAx



  ## Misc Methods #################################################################

    def smallify(self):
        self.label_im = []

    def save(self, name):
        path = self.__absPath(name)
        if os.path.isfile(path):
            os.remove(path)
        with open(path, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        
    def randomize(self):
        self.randOffset = int(self.primeRand.uniform(0, 10000))

    def setOffset(self, offset):
        self.randOffset = offset

    def __absPath(self, path):
        #Converts a relative path to an absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))   
        abs = os.path.join(script_dir, path)    
        return abs

    #def __find_nearest(self,array,value):
    #    #Returns the index of the point most similar to a given value
    #    idx = (np.abs(array-value)).argmin()
    #    return idx

    def __find_nearest(self,array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx
    def interp_rx_dat(self, rx, array):
        #Interpolates an array(rx)
        if rx < 1. : return math.nan
        locs = self.rx_raw
        return self.interp(locs, array, rx)

    def interp_frac(self, T):
        #Figures out the ionization fraction as f(T)
        locs = self.chTemps
        func = np.log10(self.chFracs)
        temp = np.log10(T)
        #print(temp)
        #plt.plot(locs, func)
        #plt.show()
        return 10**self.interp(locs, func, temp)

    def interp_f1(self, b):
        #Figures out f1 as f(b)
        locs = self.fr
        func = self.f1_raw
        return self.interp(locs, func, b)

    def interp_f2(self, b):
        #Figures out f1 as f(b)
        locs = self.fr
        func = self.f2_raw
        return self.interp(locs, func, b)

    def interp_f3(self, b):
        #Figures out f1 as f(b)
        locs = self.fr
        func = self.f3_raw
        return self.interp(locs, func, b)

    def interp_ur(self, b):
        #Figures out ur as f(b)
        locs = self.rx_raw
        func = self.ur_raw
        return self.interp(locs, func, b)

    def interp_upsilon(self, X):
        #Figures out upsilon as f(X)
        locs = self.splinedUpsX
        func = self.splinedUps
        return self.interp(locs, func, X)

    def interp(self, X, Y, K):
        #Takes in X and Y and returns linearly interpolated Y at K
        try:
            TInd = int(self.__find_nearest(X, K))
            val1 = Y[TInd]
            val2 = Y[TInd+1]
            slope = val2 - val1
            step = X[TInd+1] - X[TInd]
            discreteT = X[TInd]
            diff = K - discreteT
            diffstep = diff / step
            return val1 + diffstep*(slope)
        except: return np.nan

    def cm2km(self, var):
        return var * 1e-5

    def km2cm(self, var):
        return var * 1e5

    def ang2cm(self, var):
        return var * 1e-8

    def cm2ang(self, var):
        return var * 1e8

    def ang2km(self, var):
        return var * 1e-13

    def km2ang(self, var):
        return var * 1e13

    def ryd2ev(self,var):
        return var * 13.605693

    def ryd2erg(self,var):
        return var * 2.1798723e-11

    def ryd2ang(self,var):
        return self.cm2ang(self.c * self.hev / self.ryd2ev(var))

    def get(self, myProperty, scaling = 'None', scale = 10):
        prop = vars(self)[myProperty]
        if scaling.lower() == 'none':
            scaleProp = prop
        elif scaling.lower() == 'log':
            scaleProp = np.log10(prop)/np.log10(scale)
        elif scaling.lower() == 'root':
            scaleProp = prop**(1/scale)
        elif scaling.lower() == 'exp':
            scaleProp = prop**scale
        else: 
            print('Bad Scaling - None Used')
            scaleProp = prop
        return scaleProp

    def plot(self, property, abssisca = None, scaling = 'None', scale = 10):
        scaleProp = self.get(property, scaling, scale)

        if abssisca is not None:
            abss = self.get(abssisca)
        else: abss = np.arange(len(scaleProp))

        plt.plot(abss, scaleProp)
        plt.title(property)
        grid.maximizePlot()
        plt.show()
        
#envs Class handles creation, saving, and loading of environments
class envrs:

    #envs Class handles creation, saving, and loading of environments
    
    def __init__(self, name = '', fFile = None):
        self.name = name
        self.fFile = fFile
        self.slash = os.path.sep            
        self.savPath = os.path.normpath('../dat/envs/' + self.name)
        self.envPath = os.path.normpath('../dat/magnetograms')
        
        return
    
    def __loadEnv(self, path): 
        with open(path, 'rb') as input:
            return pickle.load(input)

    def __createEnvs(self, maxN = 1e8):
        files = glob.glob(absPath(self.envPath + self.slash + '*.sav'))
        envs = []
        ind = 0
        for file in files:
            if ind < maxN: envs.append(environment(Bfile = file, name = self.name + '_' + str(ind), fFile = self.fFile))
            ind += 1
        return envs
 
    def __saveEnvs(self, maxN = 1e8):
        ind = 0
        os.makedirs(os.path.abspath(self.savPath), exist_ok=True)
        pathname = self.savPath + self.slash + self.name
        for env in self.envs:
            if ind < maxN: env.save(os.path.abspath(pathname + '_' + str(ind) + '.env'))
            ind += 1

        infoEnv = self.envs[0]
        with open(pathname + '.txt', 'w') as output:
            output.write(time.asctime() + '\n\n')
            myVars = (infoEnv.__class__.__dict__, vars(infoEnv))
            for pile in myVars:
                for ii in sorted(pile.keys()):
                    if not callable(pile[ii]):
                        string = str(ii) + " : " + str(pile[ii]) + '\n'
                        output.write(string)
                output.write('\n\n')

    def loadEnvs(self, maxN = 1e8):
        files = glob.glob(absPath(self.savPath +os.path.normpath('/*.env')))
        self.envs = []
        ind = 0
        for file in files:
            if ind < maxN: self.envs.append(self.__loadEnv(file))
            ind += 1
        assert len(self.envs) > 0
        return self.envs
        
    def showEnvs(self, maxN = 1e8):
        try: self.envs
        except: self.loadEnvs(maxN)
        #Print all properties and values
        ind = 0
        for env in self.envs:
            if ind < maxN:
                myVars = vars(env)
                print("\nEnv {} Properties".format(ind))
                for ii in sorted(myVars.keys()):
                    if isinstance(myVars[ii], str):
                        print(ii, " : ", myVars[ii].rsplit(os.path.sep, 1)[-1])
                for ii in sorted(myVars.keys()):
                    if not isinstance(myVars[ii], (str, np.ndarray)):
                        print(ii, " : ", myVars[ii])
                envVars = vars(environment)
                for ii in sorted(envVars.keys()):
                    if isinstance(envVars[ii], (int, float)):
                        print(ii, " : ", envVars[ii])
                ind += 1 
                print("")
        return self.envs

    def Vars(self):
        return vars(self) 

    def processEnvs(self, maxN = 1e8, show = False):
        self.envs = self.__createEnvs(maxN)
        self.__saveEnvs(maxN)
        if show: plt.show(False)
        return self.envs
            

####################################################################                            
##                           Simulation                           ##
####################################################################

## Level 0: Simulates physical properties at a given coordinate
class simpoint:

    ID = 0
    useIonFrac = True    
    #Level 0: Simulates physical properties at a given coordinate
    def __init__(self, cPos = [0.1,0.1,1.5], grid = None, env = None, findT = True, pbar = None, copyPoint = None):
        #Inputs
        self.grid = grid
        self.env = env
        self.cPos = cPos
        self.pPos = self.cart2sph(self.cPos)
        self.rx = self.r2rx(self.pPos[0])
        self.zx = self.rx - 1
        self.maxInt = 0
        self.totalInt = 0
        self.bWasOn = copy.deepcopy(self.useB)
        
        

        if findT is None:
            self.findT = self.grid.findT
        else: self.findT = findT

        self.loadParams(copyPoint)
        
        #Initialization
        self.findTemp()
        self.findFootB()
        self.__findFluxAngle()
        self.findDensity()
        self.findTwave()
        self.__streamInit()
        self.findSpeeds()
        self.findQt()
        self.findUrProj()
        self.dPB()

        if pbar is not None:
            pbar.increment()
            pbar.display()

    def loadParams(self, copyPoint):
        if copyPoint is None:
            self.useWaves = simpoint.g_useWaves
            self.useWind = simpoint.g_useWind
            self.useFluxAngle = simpoint.g_useFluxAngle
            self.Bmin = simpoint.g_Bmin
        else:
            self.useWaves = copyPoint.useWaves
            self.useWind = copyPoint.useWind
            self.useFluxAngle = copyPoint.useFluxAngle
            self.Bmin = copyPoint.Bmin



  ## Temperature ######################################################################
    def findTemp(self):
        self.T = self.interp_rx_dat(self.env.T_raw)

  ## Magnets ##########################################################################
    def findFootB(self):
        #Find B
        self.__findfoot_Pos()
        #self.footB = self.env.BMap(self.foot_cPos[0], self.foot_cPos[1])[0][0]
        if self.useB:
            self.footB = self.env.BMap[self.__find_nearest(self.env.BMap_x, self.foot_cPos[0])][
                                            self.__find_nearest(self.env.BMap_y, self.foot_cPos[1])]
        else: self.footB = 0

    def __findfoot_Pos(self):
        #Find the footpoint of the field line
        self.f = self.getAreaF(self.rx)
        theta0_edge = self.env.theta0 * np.pi/180.
        theta_edge  = np.arccos(1. - self.f + self.f*np.cos(theta0_edge))
        edge_frac   = theta_edge/theta0_edge 
        coLat = self.pPos[1] /edge_frac
        self.foot_pPos = [self.env.rstar+1e-2, coLat, self.pPos[2]]
        self.foot_cPos = self.sph2cart(self.foot_pPos)

    def getAreaF(self, r):
        Hfit  = 2.2*(self.env.fmax/10.)**0.62
        return self.env.fmax + (1.-self.env.fmax)*np.exp(-((r-1.)/Hfit)**1.1)

    def __findStreamIndex(self):
        self.streamIndex = self.env.randOffset + self.env.label_im[self.__find_nearest(self.env.BMap_x, self.foot_cPos[0])][
                                            self.__find_nearest(self.env.BMap_y, self.foot_cPos[1])]

  ## Density ##########################################################################
    def findDensity(self):
        #Find the densities of the grid point
        self.densfac = self.__findDensFac()
        self.rho = self.__findRho(self.densfac) #Total density
        self.nE = 0.9*self.rho/self.env.mP #electron number density
        self.frac = self.env.interp_frac(self.T) #ion fraction

        if self.useIonFrac:
            self.nion = np.abs(0.8 * self.frac * self.env.abundance * self.rho/self.env.mP) #ion number density
        else: self.nion = self.nE

    def __findDensFac(self):
        # Find the density factor
        Bmin = self.Bmin
        #Bmin = 4.17 #This number was calculated by hand to make the pB match with/without B.
        Bmax = 50

        if self.footB < Bmin: self.B0 = Bmin
        elif self.footB > Bmax: self.B0 = Bmax
        else: self.B0 = self.footB
        dinfty = (self.B0/15)**1.18

        if False: #Plot the density factor lines
            xx = np.linspace(0,3,100)
            dd = np.linspace(0,1,5)
            for d in dd:
                plt.plot(xx, self.__densFunc(d, xx))
            plt.show()

        if self.useB:
            return self.__densFunc(dinfty, self.zx)
        else: return 1

    def __densFunc(self,d,x):
        return (0.3334 + (d - 0.3334)*0.5*(1. + np.tanh((x - 0.37)/0.26)))*3

    def __findRho(self, densfac): 
        return self.interp_rx_dat(self.env.rho_raw) * densfac  

  ## pB stuff ############################################################################

    def dPB(self):
        imp = self.cPos[2]
        r = self.pPos[0]
        u = 0.63
        R_sun = 1

        eta = np.arcsin(R_sun/r)
        mu = np.cos(eta)

        f = mu**2 / np.sqrt(1-mu**2) * np.log(mu/(1+np.sqrt(1-mu**2)))
        A = mu - mu**3
        B = 1/4 - (3/8)*mu**2 + f/2 *( (3/4)*mu**2 - 1 )
        self.dPB = self.nE * (imp / r)**2 * ( (1-u)*A + u*B )
        return self.dPB



  ## Radiative Transfer ##################################################################
    def findQt(self):
        #Chianti Stuff
        Iinf = 2.18056334613e-11 #ergs, equal to 13.61eV
        kt = self.env.KB*self.T
        dE = self.env.ryd2erg(self.env.upsInfo[2])
        upsilon = self.findUpsilon(self.T)
        self.qt = 2.172e-8*np.sqrt(Iinf/kt)*np.exp(-dE/kt)* upsilon / self.env.wi
        
    def findUpsilon(self, T):
        Eij = self.env.upsInfo[2] #Rydberg Transition energy
        K = self.env.upsInfo[6] #Transition Type
        C = self.env.upsInfo[7] #Scale Constant

        E = np.abs(T/(1.57888e5*Eij))
        if K == 1 or K == 4: X = np.log10((E+C)/C)/np.log10(E+C)
        if K == 2 or K == 3: X = E/(E+C)

        Y = self.env.interp_upsilon(X)
        if K == 1: Y = Y*np.log10(E+np.exp(1))
        if K == 3: Y = Y/(E+1)
        if K == 4: Y = Y*np.log10(E+C)

        return Y
       
    def getProfile(self):
        self.lam0 = self.env.lam0 #Angstroms
        self.deltaLam = self.lam0 / self.env.c * np.sqrt(2 * self.env.KB * self.T / self.env.mI)
        self.lamLos =  self.vLOS * self.lam0 / self.env.c

        expblock = ((self.env.lamAx - self.lam0 - self.lamLos)/(self.deltaLam))
        self.lamPhi = 1/(self.deltaLam * self.env.rtPi) * np.exp(-expblock*expblock) #Shouldn't there be twos?
        self.profile = self.nion * self.nE * self.qt * self.lamPhi

        return self.profile

    def chiantiSpectrum(self):
        a = 1
        #maxlam = self.env.lam0 + self.env.lamPm
        #minlam = self.env.lam0 - self.env.lamPm
        #lamAx = np.linspace(minlam, maxlam)
        #print(self.env.ionString)
        #s = ch.spectrum(self.T, self.nE, self.env.lamAx, 
        #        em = 1e33, doContinuum=0, ionList = self.env.ionString)
        #plt.plot(self.lamAx, s.Spectrum['intensity'][0])
        #plt.show()
        #self.ion.intensity((minlam, maxlam))
        #return self.ion.Intensity['wvl'], self.ion.Intensity['intensity']
        #self.ion.setup()
        #self.ion = copy.copy(self.env.ion)
        #self.ion.NTempDen = 1
        #self.ion.Temperature = self.T
        #self.ion.EDensity = self.nE
        #self.ion.ndens = self.ion.EDensity.size
        #self.ion.ntemp = self.ion.Temperature.size
        #self.ion.setup()
        #self.ion.ioneqOne()
        #frac = self.ion.IoneqOne

        #self.ion.upsilonDescale()
        
        ##self.ion.intensity()
        ##dist = np.abs(np.asarray(self.ion.Intensity['wvl']) - self.lam0)
        ##idx = np.argmin(dist)
        #self.qt = self.ion.Upsilon['upsilon']
        #print(len(self.ion.Wgfa['wvl']))
        pass

  ## Velocity ##########################################################################
    def findSpeeds(self, t = 0):
        #Find all of the static velocities
        self.uw = self.__findUw()
        self.vAlf = self.__findAlf()
        self.vPh = self.__findVPh()
        if self.useWaves: self.vRms = self.__findvRms()
        else: self.vRms = 0

        self.findWaveSpeeds(t)

    def findWaveSpeeds(self, t = 0):
        #Find all of the wave velocities
        self.t1 = t - self.twave + self.alfT1
        self.t2 = t - self.twave + self.alfT2
        self.alfU1 = self.vRms*self.xi1(self.t1) 
        self.alfU2 = self.vRms*self.xi2(self.t2)
        uTheta = self.alfU1 * np.sin(self.alfAngle) + self.alfU2 * np.cos(self.alfAngle)
        uPhi =   self.alfU1 * np.cos(self.alfAngle) - self.alfU2 * np.sin(self.alfAngle)
        pU = [self.uw, uTheta, uPhi]

        if self.useFluxAngle:
            pU = self.fluxAngleOffset(pU, self.delta)

        self.updateVelocities(pU)

    def updateVelocities(self, pU):
        self.ur, self.uTheta, self.uPhi = pU
        self.pU = pU
        self.cU = self.__findCU(pU) 
        self.ux, self.uy, self.uz = self.cU
        self.vLOS = self.__findVLOS()  

    def __findFluxAngle(self):
        dr = 1e-4
        r1 = self.rx
        r2 = r1 + dr

        thetar1 = np.arccos(1 - self.getAreaF(r1)*(1-np.cos(self.foot_pPos[1])))
        thetar2 = np.arccos(1 - self.getAreaF(r2)*(1-np.cos(self.foot_pPos[1])))
        dtheta = thetar1 - thetar2

        self.delta = -np.arctan2(r1 * dtheta , dr)
        self.dangle = self.pPos[1] + self.delta
        #self.dx = np.cos(self.delta)
        #self.dy = np.sin(self.delta)

    def fluxAngleOffset(self, pU, delta):
        ur = pU[0]
        utheta = pU[1]

        newUr =        ur * np.cos(delta) - utheta * np.sin(delta) 
        newTheta = utheta * np.cos(delta) + ur * np.sin(delta)

        newPU = [newUr, newTheta, pU[2]]
        return newPU

    def findUrProj(self):
        theta = self.pPos[1]

        if self.useFluxAngle:
            delta = self.delta
        else: delta = 0

        self.urProj =  0.5*(np.sin(theta + delta) * self.uw)**2 * self.rho**2
        self.rmsProj = (np.cos(theta + delta) * self.vRms)**2 * self.rho**2
        self.temProj = self.T * self.rho**2

    def __streamInit(self):
        self.__findStreamIndex()
        self.env.streamRand.seed(int(self.streamIndex))
        thisRand = self.env.streamRand.random_sample(3)
        self.alfT1 =  thisRand[0] * self.env.last_xi1_t
        self.alfT2 =  thisRand[1] * self.env.last_xi2_t
        self.alfAngle = thisRand[2] * 2 * np.pi
        self.omega = 0.01

    def __findUw(self):
        #Wind Velocity
        #return (1.7798e13) * self.fmax / (self.num_den * self.rx * self.rx * self.f)
        if self.useWind:
            uw = self.interp_rx_dat(self.env.ur_raw) / self.densfac
        else:
            uw = 0
        return uw

    def __findAlf(self):
        #Alfven Velocity
        #return 10.0 / (self.f * self.rx * self.rx * np.sqrt(4.*np.pi*self.rho))
        return self.interp_rx_dat(self.env.vAlf_raw) / np.sqrt(self.densfac)

    def __findVPh(self):
        #Phase Velocity
        return self.vAlf + self.uw 
    
    def __findvRms(self):
        #RMS Velocity
        S0 = 7.0e5 #* np.sqrt(1/2)
        return np.sqrt(S0*self.vAlf/((self.vPh)**2 * 
            self.rx**2*self.f*self.rho))

    def __findVLOS(self, nGrad = None):
        if nGrad is not None: self.nGrad = nGrad
        else: self.nGrad = self.grid.ngrad
        self.vLOS = np.dot(self.nGrad, self.cU)
        #self.vLOS = self.alfU1 #FOR TESTING
        return self.vLOS

    def __findVLOS2(self, vel, nGrad = None):
        if nGrad is not None: self.nGrad = nGrad
        else: self.nGrad = self.grid.ngrad
        vLOS2 = np.dot(self.nGrad, vel)
        #print(self.vLOS2)
        return vLOS2

    def __findVPerp2(self, vel, nGrad = None):
        if nGrad is not None: self.nGrad = nGrad
        else: self.nGrad = self.grid.ngrad
        vPerp2 = np.cross(self.nGrad, vel)
        #print(self.vLOS2)
        return vPerp2

    def __findCU(self, pU):
        #Finds the cartesian velocity components
        [ur, uTheta, uPhi] = pU
        ux = -np.cos(self.pPos[2])*(ur*np.sin(self.pPos[1]) + uTheta*np.cos(self.pPos[1])) - np.sin(self.pPos[2])*uPhi
        uy = -np.sin(self.pPos[2])*(ur*np.sin(self.pPos[1]) + uTheta*np.cos(self.pPos[1])) - np.cos(self.pPos[2])*uPhi
        uz = ur*np.cos(self.pPos[1]) - uTheta*np.sin(self.pPos[1])
        return [ux, uy, uz]
    
  ## Time Dependence #######################################################################    
    def findTwave(self):
        #Finds the wave travel time to this point
        #Approximate Version
        twave_min = 161.4 * (self.rx**1.423 - 1.0)**0.702741
        self.twave_fit = twave_min * self.densfac
        
        if self.findT:
            #Real Version 
            radial = grid.sightline(self.foot_cPos, self.cPos)
            N = 10
            wtime = 0
            rLine = radial.cGrid(N)
            for cPos in rLine:
                point = simpoint(cPos, findT = False, grid = radial, env = self.env)
                wtime += (1/point.vPh) / N
            self.twave = wtime * self.r2rx(radial.norm) * 69.63e9 #radius of sun in cm
            if self.twave < 0: self.twave = -256
        else:
            self.twave = self.twave_fit  
        if not self.twave_fit == 0:
            self.twave_rat = self.twave/self.twave_fit
        else: self.twave_rat = np.nan

        
    def setTime(self,t = 0):
        #Updates velocities to input time
        self.findWaveSpeeds(t)
        self.__findVLOS()

    def xi1(self, t):
        #Returns xi1(t)
        if math.isnan(t):
            return math.nan
        else:
            t_int = int(t % self.env.last_xi1_t)
            xi1 = self.env.xi1_raw[t_int]
            xi2 = self.env.xi1_raw[t_int+1]
            return xi1+( (t%self.env.last_xi1_t) - t_int )*(xi2-xi1)

    def xi2(self, t):
        #Returns xi2(t)
        if math.isnan(t):
            return math.nan
        else:
            t_int = int(t % self.env.last_xi2_t)
            xi1 = self.env.xi2_raw[t_int]
            xi2 = self.env.xi2_raw[t_int+1]
            return xi1+( (t%self.env.last_xi2_t) - t_int )*(xi2-xi1)

  ## Misc Methods ##########################################################################
    #def __find_nearest(self,array,value):
    #    #Returns the index of the point most similar to a given value
    #    idx = (np.abs(array-value)).argmin()
    #    return idx

    def __find_nearest(self,array,value):
        idx = np.searchsorted(array, value, side="left")
        if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
            return idx-1
        else:
            return idx

    def interp_rx_dat(self, array):
        #Interpolates an array(rx)
        if self.rx < self.env.rx_raw[0] : return math.nan
        rxInd = np.int(np.floor(self.__find_nearest(self.env.rx_raw, self.rx)))
        val1 = array[rxInd]
        val2 = array[rxInd+1]
        slope = val2 - val1
        step = self.env.rx_raw[rxInd+1] - self.env.rx_raw[rxInd]
        discreteRx = self.env.rx_raw[rxInd]
        diff = self.rx - discreteRx
        diffstep = diff / step
        return val1+ diffstep*(slope)

    def r2rx(self, r):
        return r/self.env.rstar
                          
    def sph2cart(self, sph):
        #Change coordinate systems
        rho, theta, phi = sph[:]
        x = np.array(rho)*np.sin(np.array(theta))*np.cos(np.array(phi))
        y = np.array(rho)*np.sin(np.array(theta))*np.sin(np.array(phi))
        z = np.array(rho)*np.cos(np.array(theta))
        return [x, y, z]
        
    def cart2sph(self, cart):  
        #Change coordinate systems
        x,y,z = cart[:]
        if x == 0: x = 1e-8
        if y == 0: y = 1e-8
        rho = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arccos(z/rho)
        phi = np.arctan2(y, x)

        return [rho, theta, phi]
        
    def __absPath(self, path):
        #Converts a absative path to an absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))   
        abs = os.path.join(script_dir, path)    
        return abs
            
    def show(self):
        #Print all properties and values
        myVars = vars(self)
        print("\nSimpoint Properties")
        for ii in sorted(myVars.keys()):
            print(ii, " : ", myVars[ii])

    def Vars(self):
        return vars(self) 


## Level 1: Initializes many Simpoints into a Simulation
class simulate: 
    #Level 1: Initializes many Simpoints into a Simulation
    def __init__(self, gridObj, envObj, N = None, iL = None, findT = None, printOut = False, timeAx = [0], getProf = False):
        self.print = printOut
        #if self.print: print("Initializing Simulation...")
        self.grid  = gridObj
        self.findT = findT
        self.env = envObj
        self.timeAx = timeAx
        self.profile = None
        self.adapt = False

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.root = self.rank == 0
        self.size = self.comm.Get_size()
        

        if findT is None:
            self.findT = self.grid.findT
        else: self.findT = findT

        if iL is None: self.iL = self.grid.iL
        else: self.iL = iL

        if N is None: self.N = self.grid.default_N
        else: self.N = N
        
        if type(self.N) is list or type(self.N) is tuple: 
            self.adapt = True
            self.grid.setMaxStep(1/self.N[0])
            self.grid.setMinStep(1/self.N[1])
        else: self.grid.setN(self.N)

        self.simulate_now()
        if getProf: self.lineProfile()
        #print(self.pB)
        #sys.stdout.flush()
        

    def simulate_now(self):
  
        if type(self.grid) is grid.plane: 
            doBar = True
            self.adapt = False
            self.Npoints = self.grid.Npoints
        else: doBar = False
        
        if doBar and self.print: bar = pb.ProgressBar(self.Npoints)
        self.sims = []
        self.steps = []
        self.pData = []
        #if self.print: print("\nBeginning Simulation...")

        t = time.time()
        stepInd = 0
        rhoSum = 0
        tol = 0.67
        for cPos, step in self.grid: 

            thisPoint = simpoint(cPos, self.grid, self.env, self.findT) 

            if self.adapt:
                #Adaptive Mesh
                thisDens = thisPoint.densfac

                if (thisDens > tol) and self.grid.backflag:
                    self.grid.back()
                    self.grid.set2minStep()
                    self.grid.backflag = False
                    continue
                if thisDens <= tol:
                    self.grid.incStep(1.5)
                    self.grid.backflag = True

            stepInd += 1
            
            self.sims.append(thisPoint)
            self.steps.append(step)
            self.pData.append(thisPoint.Vars())
            
            if doBar and self.print: 
                bar.increment()
                bar.display()
        self.cumSteps = np.cumsum(self.steps)
        if doBar and self.print: bar.display(force = True)
        #if self.print: print('Elapsed Time: ' + str(time.time() - t))

        self.Npoints = len(self.sims)
        if type(self.grid) is grid.sightline:
            self.shape = [self.Npoints, 1] 
        else: 
            self.shape = self.grid.shape
        self.shape2 = [self.shape[0], self.shape[1], -1]

    def get(self, myProperty, dim = None, scaling = 'None', scale = 10):
        propp = np.array([x[myProperty] for x in self.pData])
        prop = propp.reshape(self.shape2)
        if not dim is None: prop = prop[:,:,dim]
        prop = prop.reshape(self.shape)
        if scaling.lower() == 'none':
            scaleProp = prop
        elif scaling.lower() == 'log':
            scaleProp = np.log10(prop)/np.log10(scale)
        elif scaling.lower() == 'root':
            scaleProp = prop**(1/scale)
        elif scaling.lower() == 'exp':
            scaleProp = prop**scale
        else: 
            print('Bad Scaling - None Used')
            scaleProp = prop
        datSum = sum((v for v in scaleProp.ravel() if not math.isnan(v)))
        return scaleProp, datSum

    def plot(self, property, dim = None, scaling = 'None', scale = 10, cmap = 'jet', axes = True, center = False, linestyle = 'b'):
        scaleProp, datSum = self.get(property, dim, scaling, scale)
        self.fig, ax = self.grid.plot(iL = self.iL)
        if type(self.grid) is grid.sightline:
            #Line Plot
            im = ax.plot(self.cumSteps, scaleProp, linestyle)
            if axes:
                ax.axhline(0, color = 'k')
                ax.axvline(0.5, color = 'k')
                ax.set_xlim([0,1])
            datSum = datSum / self.N

        elif type(self.grid) is grid.plane:
            #Image Plot
            if center:
                v = np.nanmax(np.abs(scaleProp))
                im = ax.imshow(scaleProp, interpolation='none', cmap = cmap, vmin = -v, vmax = v)
            else:
                im = ax.imshow(scaleProp, interpolation='none', cmap = cmap)
            self.fig.subplots_adjust(right=0.89)
            cbar_ax = self.fig.add_axes([0.91, 0.10, 0.03, 0.8], autoscaley_on = True)
            self.fig.colorbar(im, cax=cbar_ax)
            datSum = datSum / self.N ** 2
        else: 
            print("Invalid Grid")
            return

        if dim is None:
            ax.set_title(property + ", scaling = " + scaling + ', sum = {}'.format(datSum))
        else:
            ax.set_title(property + ", dim = " + dim.__str__() + ", scaling = " + scaling + ', sum = {}'.format(datSum))

        grid.maximizePlot()
        plt.show()

    def quiverPlot(self):
        dx, datSum = self.get('dx')
        dy, datSum = self.get('dy')
        delta, datSum = self.get('delta')
        plt.quiver(dx, dy, scale = 50, color = 'w')
        rho, datsum = self.get('rho', scaling = 'log')
        plt.imshow(delta, interpolation = "None")
        plt.show()

    def plot2(self, p1, p2, scaling1 = 'None', scaling2 = 'None', dim1 = None, dim2 = None, axes = True):
        scaleProp1, datSum1 = self.get(p1, dim1, scaling1)
        scaleProp2, datSum2 = self.get(p2, dim2, scaling2)
        datSum = datSum1
        self.fig, ax = self.grid.plot(iL = self.iL)
        if type(self.grid) is grid.sightline:
            #Line Plot
            im = ax.plot(self.cumSteps, scaleProp1, label = p1)
            im = ax.plot(self.cumSteps, scaleProp2, label = p2)
            if axes:
                ax.axhline(0, color = 'k')
                ax.axvline(0.5, color = 'k')
                ax.set_xlim([0,1])
                ax.legend()
            datSum = datSum / self.N
        elif type(self.grid) is grid.plane:
            #Image Plot
            im = ax.imshow(scaleProp, interpolation='none', cmap = cmap)
            self.fig.subplots_adjust(right=0.89)
            cbar_ax = self.fig.add_axes([0.91, 0.10, 0.03, 0.8], autoscaley_on = True)
            self.fig.colorbar(im, cax=cbar_ax)
            datSum = datSum / self.N ** 2
        else: 
            print("Invalid Grid")
            return

        if dim1 is None:
            ax.set_title(p1 + ", scaling = " + scaling1 + ', sum = {}'.format(datSum1) + '\n' +
                            p2 + ", scaling = " + scaling2 + ', sum = {}'.format(datSum2))
        else:
            ax.set_title(p1 + ", dim = " + dim1.__str__() + ", scaling = " + scaling + ', sum = {}'.format(datSum))

        grid.maximizePlot()
        plt.show()

    def compare(self, p1, p2, scaling1 = 'None', scaling2 = 'None', dim1 = None, dim2 = None, center = False):
        scaleprop = []
        scaleprop.append(self.get(p1, dim1, scaling1)[0])
        scaleprop.append(self.get(p2, dim2, scaling2)[0])
        fig, ax = self.grid.plot(iL = self.iL)
        fig.subplots_adjust(right=0.89)
        cbar_ax = fig.add_axes([0.91, 0.10, 0.03, 0.8], autoscaley_on = True)
        if center:
            vmax = np.nanmax(np.abs(scaleprop[0]))
            vmin = - vmax
        else: vmax = vmin = None

        global cur_plot
        cur_plot = 0

        def plot1():
            im = ax.imshow(scaleprop[0], interpolation = 'none', vmin = vmin, vmax = vmax)
            ax.set_title(p1)
            fig.colorbar(im, cax=cbar_ax)
            fig.canvas.draw()

        def plot2():
            im = ax.imshow(scaleprop[1], interpolation = 'none', vmin = vmin, vmax = vmax)
            ax.set_title(p2)
            fig.colorbar(im, cax=cbar_ax)
            fig.canvas.draw()

        plots = [plot1, plot2]

        def onKeyDown(event):
            global cur_plot
            cur_plot = 1 - cur_plot
            plots[cur_plot]()

        cid = fig.canvas.mpl_connect('key_press_event', onKeyDown)
       
        grid.maximizePlot()
        plt.show()    

    def show(self):
        #Print all properties and values
        myVars = vars(self.sims[0])
        print("\nSimpoint Properties")
        for ii in sorted(myVars.keys()):
            print(ii, " : ", myVars[ii])

    def Vars(self):
        #Returns the vars of the simpoints
        return self.sims[0].Vars()

    def Keys(self):
        #Returns the keys of the simpoints
        return self.sims[0].Vars().keys()

    ####################################################################

    #def getProfile(self):
    #    return self.lineProfile()

    def plotProfile(self):
        if self.profile is None: self.lineProfile()
        plt.plot(self.env.lamAx, (self.profile))
        plt.title("Line Profile")
        plt.ylabel("Intensity")
        plt.xlabel("Wavelenght (A)")
        plt.yscale('log')
        plt.show()

    def setIntLam(self, lam):
        for point in self.sims:
            point.findIntensity(self.env.lam0, lam)

    def randomizeTime(self):
        if self.timeAx[0] == 'rand':
            self.env.timeRand.seed(int(self.env.primeSeed + self.rank))
            self.timeAx = [self.env.timeRand.randint(self.env.tmax)]#put random num gen here
            

    def lineProfile(self):
        #Get a line profile integrated over time
        profile = np.zeros_like(self.env.lamAx)
        urProj = 0
        rmsProj = 0
        temProj = 0
        rho2 = 0
        pB = 0
        self.randomizeTime()
        if self.print and self.root: 
            print('\nGenerating Profile...')
            bar = pb.ProgressBar(len(self.sims) * len(self.timeAx))
        for point, step in zip(self.sims, self.steps):
            for tt in self.timeAx:
                point.setTime(tt)

                profile += point.getProfile() * step
                pB += point.dPB * step
                urProj += point.urProj * step
                rmsProj += point.rmsProj * step
                temProj += point.temProj * step
                rho2 += point.rho**2 * step

                if self.print and self.root:
                    bar.increment()
                    bar.display()
        self.profile = profile
        self.pB = pB
        self.urProj = np.sqrt(urProj/rho2)
        self.rmsProj = np.sqrt(rmsProj/rho2)
        self.temProj = (temProj/rho2)

        #plt.plot(profile)
        #plt.show()
        if self.print and self.root: bar.display(True)
        return self.profile



## Time Dependence ######################################################
    def setTime(self, tt = 0):
        for point in self.sims:
            point.setTime(tt)

    def makeSAxis(self):
        loc = 0
        ind = 0
        self.sAxis = np.zeros_like(self.steps)
        for step in self.steps:
            loc += step
            self.sAxis[ind] = loc
            ind += 1
        self.sAxisList = self.sAxis.tolist()
        
    def peekLamTime(self, lam0 = 1000, lam = 1000, t = 0):
        self.makeSAxis()
        intensity = np.zeros_like(self.sims)
        pInd = 0
        for point in self.sims:
            point.setTime(t)
            intensity[pInd] = point.findIntensity(lam0, lam)
            pInd += 1
        plt.plot(self.sAxis, intensity, '-o')
        plt.show()

    def evolveLine(self, tn = 150, t0 = 0, t1 = 4000):
        #Get the line profile over time and store in LineArray
        print('Timestepping...')
        
        #self.lam0 = lam0
        self.times = np.linspace(t0, t1, tn)
        #self.makeLamAxis(self, Ln, lam0, lamPm)
        self.lineArray = np.zeros((tn, self.env.Ln))

        bar = pb.ProgressBar(len(self.times))
        timeInd = 0
        for tt in self.times:
            for point in self.sims:
                point.setTime(tt)
            self.lineArray[timeInd][:] = self.lineProfile()
            bar.increment()
            bar.display()  
            timeInd += 1       
        bar.display(force = True)

        self.lineList = self.lineArray.tolist()


        self.__fitGaussians_t()
        self.__findMoments_t()
        self.plotLineArray_t()

    def __findMoments_t(self):
        #Find the moments of each line in lineList
        self.maxMoment = 3
        self.moment = []
        lineInd = 0
        for mm in np.arange(self.maxMoment):
            self.moment.append(np.zeros_like(self.times))

        for line in self.lineList:
            for mm in np.arange(self.maxMoment):
                self.moment[mm][lineInd] = np.dot(line, self.env.lamAx**mm)
            lineInd += 1

        self.__findMomentStats_t()
        #self.plotMoments_t()
        return

    def __findMomentStats_t(self):
        self.power = self.moment[0]
        self.centroid = self.moment[1] / self.moment[0]
        self.sigma = np.sqrt(self.moment[2]/self.moment[0] - (self.moment[1]/self.moment[0])**2)
        #self.plotMomentStats_t()

    def plotMomentStats_t(self):
        f, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
        f.suptitle('Moments Method')
        ax1.plot(self.times, self.power)
        ax1.set_title('0th Moment')
        ax1.get_yaxis().get_major_formatter().set_useOffset(False)
        ax1.get_yaxis().get_major_formatter().set_scientific(True)

        ax2.plot(self.times, self.centroid)
        ax2.set_title('Centroid')
        ax2.set_ylabel('Angstroms')

        ax3.plot(self.times, self.sigma)
        ax3.set_ylabel('Angstroms')
        ax3.set_title('Std')
        ax3.set_xlabel('Time (s)')
        plt.show(False)

    def plotMoments_t(self):
        f, axArray = plt.subplots(self.maxMoment, 1, sharex=True)
        mm = 0
        for ax in axArray:
            ax.plot(self.times, self.moment[mm])
            ax.set_title(str(mm)+" Moment")
            ax.set_ylabel('Angstroms')
            mm += 1
        ax.set_xlabel('Time (s)')
        plt.show(False)

    def plotLineArray_t(self):
        ## Plot the lineArray
        self.fig, ax = self.grid.plot(iL = self.iL)
        #print('')
        #print(np.shape(self.lineArray))
        im = ax.pcolormesh(self.env.lamAx.astype('float32'), self.times, self.lineArray)
        #ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.set_xlabel('Angstroms')
        ax.set_ylabel('Time (s)')
        self.fig.subplots_adjust(right=0.89)
        cbar_ax = self.fig.add_axes([0.91, 0.10, 0.03, 0.8], autoscaley_on = True)
        self.fig.colorbar(im, cax=cbar_ax)

        #Plot the Centroid vs Time
        self.fig.subplots_adjust(right=0.7)
        cent_ax = self.fig.add_axes([0.74, 0.10, 0.15, 0.8], autoscaley_on = True)
        cent_ax.set_xlabel('Centroid')
        cent_ax.plot(self.centroid, self.times)
        cent_ax.plot(np.ones_like(self.times)*self.env.lam0, self.times)
        #cent_ax.set_xlim([np.min(self.env.lamAx), np.max(self.env.lamAx)])
        cent_ax.xaxis.get_major_formatter().set_useOffset(False)
        max_xticks = 4
        xloc = plt.MaxNLocator(max_xticks)
        cent_ax.xaxis.set_major_locator(xloc)

        grid.maximizePlot()
        plt.show(False)

    def __fitGaussians_t(self):
        self.amp = np.zeros_like(self.times)
        self.mu = np.zeros_like(self.times)
        self.std = np.zeros_like(self.times)
        self.area = np.zeros_like(self.times)

        def gauss_function(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        lInd = 0
        for line in self.lineList:
            sig0 = sum((self.env.lamAx - self.env.lam0)**2)/len(line)
            amp0 = np.max(line)
            popt, pcov = curve_fit(gauss_function, self.env.lamAx, line, p0 = [amp0, self.env.lam0, sig0])
            self.amp[lInd] = popt[0]
            self.mu[lInd] = popt[1] - self.env.lam0
            self.std[lInd] = popt[2] 
            self.area[lInd] = popt[0] * popt[2]  

            ## Plot each line fit
            #plt.plot(self.lamAx, gauss_function(self.lamAx, amp[lInd], mu[lInd], std[lInd]))
            #plt.plot(self.lamAx,  lineList[lInd])
            #plt.show()
            lInd += 1

        #self.plotGaussStats_t()

    def plotGaussStats_t(self):
        f, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, sharex=True)
        f.suptitle('Gaussian method')
        ax1.plot(self.times, self.amp)
        ax1.set_title('Amplitude')
        ax2.plot(self.times, self.mu)
        ax2.set_title('Mean')
        ax2.set_ylabel('Angstroms')
        ax3.plot(self.times, self.std)
        ax3.set_ylabel('Angstroms')
        ax3.set_title('Std')
        ax4.plot(self.times, self.area)
        ax4.set_title('Area')
        ax4.set_xlabel('Time (s)')
        plt.show(False)


## Level 2: Initializes many simulations (MPI Enabled) for statistics
class multisim:
    #Level 2: Initializes many simulations
    def __init__(self, batch, envs, N = 1000, findT = None, printOut = False, printSim = False, timeAx = [0]):
        self.print = printOut
        self.printSim = printSim
        self.timeAx = timeAx
        self.gridLabels = batch[1]
        self.oneBatch = batch[0]
        self.batch = []
        self.envInd = []
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.root = self.rank == 0
        self.size = self.comm.Get_size()

        if type(envs) is list or type(envs) is tuple: 
            self.envs = envs
            self.Nenv = len(self.envs)
            import copy
            for nn in np.arange(self.Nenv):
                self.batch.extend(copy.deepcopy(self.oneBatch))
                self.envInd.extend([nn] * len(self.oneBatch))
        else: 
            self.envs = [envs]
            self.batch = self.oneBatch
            self.envInd = [0] * len(self.oneBatch)
        
        self.N = N
        self.findT = findT
        if self.comm.size > 1:
            self.MPI_init_masterslave()
        else: self.init_serial()
        #self.findProfiles()

    def MPI_init_fixed(self):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.root = self.rank == 0
        self.size = self.comm.Get_size()

        if self.root and self.print: 
            print('Running MultiSim: ' + time.asctime())
            t = time.time() #Print Stuff

        gridList = self.__seperate(self.batch, self.size)
        self.gridList = gridList[self.rank]  
        #print("Rank: " + str(self.rank) + " ChunkSize: " + str(len(self.gridList))) 
        #sys.stdout.flush()   
        envIndList = self.__seperate(self.envInd, self.size)
        self.envIndList = envIndList[self.rank]

        if self.root and self.print: 
            #print(self.oneBatch)
            print('Nenv = ' + str(self.Nenv), end = '; ')
            print('Lines\Env = ' + str(len(self.oneBatch)), end = '; ')
            print('JobSize = ' + str(len(self.batch)))

            print('PoolSize = ' + str(self.size), end = '; ')
            print('ChunkSize = ' + str(len(self.gridList)), end = '; ') 
            print('Short Cores: ' + str(self.size * len(self.gridList) - len(self.batch)))#Print Stuff

            bar = pb.ProgressBar(len(self.gridList))
            bar.display()

        #self.simList = []
        profiles = []

        for grd, envInd in zip(self.gridList, self.envIndList):
            self.envs[envInd].randomize()
            simulation = simulate(grd, self.envs[envInd], self.N, findT = self.findT, timeAx = self.timeAx, printOut = self.printSim)
            #self.simList.append(simulation)
            profiles.append(simulation.getProfile())
            #elapsed = time.time() - t
            if self.root and self.print:
                #print('')
                #print(elapsed)
                bar.increment()
                bar.display()
        if self.root and self.print: bar.display(force = True)

        profilebin = self.comm.gather(profiles, root = 0)

        if self.root:
            self.profiles = []
            for core in profilebin:
                for line in core:
                    self.profiles.append(line)
            print("Total Lines: " + str(len(self.profiles)))

    def MPI_init_masterslave(self):


        if self.root and self.print: 
            print('Running MultiSim: ' + time.asctime())
            t = time.time() #Print Stuff
            print('Nenv = ' + str(self.Nenv), end = '; ')
            print('Lines\Env = ' + str(len(self.oneBatch)), end = '; ')
            print('JobSize = ' + str(len(self.batch)))

            print('PoolSize = ' + str(self.size), end = '; ')
            #print('ChunkSize = ' + str(len(self.gridList)), end = '; ') 
            #print('Short Cores: ' + str(self.size * len(self.gridList) - len(self.batch)))#Print Stuff

            self.Bar = pb.ProgressBar(len(self.batch))
            self.Bar.display()
        else: self.Bar = None

        work = [[bat,env] for bat,env in zip(self.batch, self.envInd)]
        self.sims = self.poolMPI(work, self.mpi_sim, bar)
        self.collectVars(self.sims)

        if self.destroySims and self.root: self.sims = self.sims[0:1]
        #if self.root:
        #    #self.profiles, self.pBs = zip(*all_work)

        #else: self.profiles, self.pBs = None, None


    def collectVars(self, sims):
        if self.root:
            self.profiles = []
            self.pBs = []
            self.urProjs = []
            self.rmsProjs = []
            self.temProjs = []
            for simulation in sims:
                self.profiles.append(simulation.profile)
                self.pBs.append(simulation.pB)
                self.urProjs.append(simulation.urProj)
                self.rmsProjs.append(simulation.rmsProj)
                self.temProjs.append(simulation.temProj)
            if self.print:
                self.Bar.display(force = True)
                print("Total Lines: " + str(len(self.profiles)))
        else:
                self.profiles = None
                self.pBs = None
                self.urProjs = None
                self.rmsProjs = None
                self.temProjs = None

    def mpi_sim(self, data):
        grd, envInd = data
        self.envs[envInd].randomize()
        simulation = simulate(grd, self.envs[envInd], self.N, findT = self.findT, timeAx = self.timeAx, printOut = self.printSim, getProf = True)
        #profile = simulation.lineProfile()
        #pB = simulation.pB
        return simulation

    def getLineArray(self):
        return np.asarray(self.profiles)
            
    def __seperate(self, list, N):
        #Breaks a list up into chunks
        import copy
        newList = copy.deepcopy(list)
        Nlist = len(newList)
        chunkSize = float(Nlist/N)
        chunks = [ [] for _ in range(N)] 
        chunkSizeInt = int(np.floor(chunkSize))

        if chunkSize < 1:
            remainder = Nlist
            chunkSizeInt = 0
            if self.root: print(' **Note: All PEs not being utilized** ')
        else:
            remainder = Nlist - N * chunkSizeInt

        for NN in np.arange(N):
            thisLen = chunkSizeInt
            if remainder > 0:
                thisLen += 1
                remainder -= 1
            for nn in np.arange(thisLen):
                chunks[NN].extend([newList.pop(0)])              
        return chunks

    def init_serial(self):
        #Serial Version

        work = [[bat,env] for bat,env in zip(self.batch, self.envInd)]
        self.sims = []
        self.Bar = pb.ProgressBar(len(work))

        for line in work:
            self.sims.append(self.mpi_sim(line))
            self.Bar.increment()
            self.Bar.display()

        self.collectVars(self.sims)

    def plotLines(self):
        axes = self.oneBatch[0].quadAxOnly()
        for line in self.oneBatch:  
            line.plot(axes = axes)   
        plt.show()   

    def dict(self):
        """
        Determine existing fields
        """
        return {field:getattr(self, field) 
                for field in dir(self) 
                if field.upper() == field 
                and not field.startswith('_')}

    def getSmall(self):
        profiles = self.profiles 

        for field in self.dict():
            delattr(self, field)

        self.profiles = profiles
        print(vars(self))
        return self

 
    def master(self, wi):
        WORKTAG = 0
        DIETAG = 1
        all_data = []
        size = MPI.COMM_WORLD.Get_size()
        current_work = self.__Work__(wi) 
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        for i in range(1, size): 
            anext = current_work.get_next_item() 
            if not anext: break
            comm.send(anext, dest=i, tag=WORKTAG)
 
        while 1:
            anext = current_work.get_next_item()
            if not anext: break
            data = comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            all_data.append(data)
            try:
                self.Bar.increment() 
                self.Bar.display()
            except: pass
            comm.send(anext, dest=status.Get_source(), tag=WORKTAG)

 
        for i in range(1,size):
            data = comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            all_data.append(data)
            try:
                bar.increment() 
                bar.display()
            except: pass
    
        for i in range(1,size):
            comm.send(None, dest=i, tag=DIETAG)
     
        return all_data    
    
    def slave(self, do_work):
        comm = MPI.COMM_WORLD
        status = MPI.Status()
        while 1:
            data = comm.recv(None, source=0, tag=MPI.ANY_TAG, status=status)
            if status.Get_tag(): break
            comm.send(do_work(data), dest=0)
    
    def poolMPI(self, work_list, do_work):
        rank = MPI.COMM_WORLD.Get_rank()
        name = MPI.Get_processor_name()
        size = MPI.COMM_WORLD.Get_size() 
    
        if rank == 0:
            all_dat = self.master(work_list)
            return all_dat
        else:
            self.slave(do_work)
            return None

    class __Work__():
        def __init__(self, work_items):
            self.work_items = work_items[:] 
 
        def get_next_item(self):
            if len(self.work_items) == 0:
                return None
            return self.work_items.pop()      


## Level 3: Initializes many Multisims, varying a parameter. Does statistics. Saves and loads Batches.
class batch:
    def __init__(self, batchname):
        self.batchName = batchname
        
    #Handles loading and running of batches
    def loadBatch(self):
        print("Loading Batch...")
        slash = os.path.sep
        batchPath = '..' + slash + 'dat' + slash + 'batches' + slash + self.batchName + '.batch'
        absPth = absPath(batchPath)
        try:
            with open(absPth, 'rb') as input:
                return pickle.load(input)
        except:
            sys.exit('Batch Not found')

    def restartBatch(self):
        myBatch = self.loadBatch()
        myBatch.findRank()
        myBatch.simulate_now()
        return myBatch

    def analyzeBatch(self, width = False):
        myBatch = self.loadBatch()
        myBatch.findRank()
        myBatch.doStats(width)
        return myBatch

class batchjob:

    #Requires labels, xlabel, batch, N
    def __init__(self, envs):
    
        if type(envs) is list or type(envs) is tuple: 
            self.envs = envs
        else: self.envs = [envs]
        self.env = self.envs[0]
        self.firstRunEver = True
        self.complete = False
        comm = MPI.COMM_WORLD
        self.root = comm.rank == 0

        try:
            self.statType = copy.deepcopy(batchjob.g_statType)
            self.usePsf = copy.deepcopy(batchjob.g_usePsf)
        except: pass

        self.simulate_now()
        
        if self.root:
            self.finish()

    def simulate_now(self):
        comm = MPI.COMM_WORLD
        if self.root and self.print:
            print('\nCoronaSim!')
            print('Written by Chris Gilbert')
            print('-------------------------\n')
            print("Simulating Impacts: \n{}".format(self.labels))


        if self.firstRunEver:
            self.count = 0
            self.batch = self.fullBatch
            self.initLists()
            if self.root and self.print: 
                self.bar = pb.ProgressBar(len(self.labels))


            self.doLabels = self.labels.tolist()
            self.doneLabels = []

        if self.root and self.print and self.printMulti: 
            print('\nBatch Progress: '+ str(self.batchName))
            self.bar.display()

        

        while len(self.doLabels) > 0:
            ind = self.doLabels.pop(0)
            self.doneLabels.append(ind)
            thisBatch = self.batch.pop(0)
            try:
                self.count += 1
            except:
                self.count = 1
            if self.root and self.printMulti: print('\n\n\n--' + self.xlabel +' = ' + str(ind) + ': [' + str(self.count) + '/' + str(self.Nb) + ']--') 

            
            
            thisSim = multisim(thisBatch, self.envs, self.N, printOut = self.printMulti, printSim = self.printSim, timeAx = self.timeAx)
            comm.barrier()

            if self.root:
                self.collectVars(thisSim)

                if self.print:
                    if self.printMulti: print('\nBatch Progress: '+ str(self.batchName))
                    self.bar.increment()
                    self.bar.display(True)
                if self.firstRunEver: self.setFirstPoint()
                self.firstRunEver = False
                self.save(printout = True)


    def finish(self):
        if self.root:
            #self.__findBatchStats()
            
            self.makeVrms()
            self.doOnePB()

            if self.complete is False:
                self.completeTime = time.asctime()
            self.complete = True

            if self.print: 
                print('\nBatch Complete: '+ str(self.batchName))
                try: print(self.completeTime)
                except: print('')
            if self.fName is not None: 
                self.calcFfiles()
            self.save(printout = True)

    def setFirstPoint(self):
        self.copyPoint = self.sims[0].sims[0].sims[0]

    def initLists(self):
            self.sims = []
            self.profiless = []
            self.pBss = []
            self.urProjss = []
            self.rmsProjss = []
            self.temProjss = []

    def collectVars(self, thisSim):
        self.sims.append(thisSim)
        self.profiless.append(thisSim.profiles)
        self.pBss.append(thisSim.pBs)
        self.urProjss.append(thisSim.urProjs)
        self.rmsProjss.append(thisSim.rmsProjs)
        self.temProjss.append(thisSim.temProjs)

    def findRank(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.root = self.rank == 0

    def doStats(self, width):
        if not self.lockFlags:
            try:
                self.statType = copy.deepcopy(batchjob.g_statType)
                self.usePsf = copy.deepcopy(batchjob.g_usePsf)
            except: pass
        self.makePSF(self.env.psfSig)
        self.__findBatchStats()
        self.makeVrms()
        self.plot(width)
        #self.doPB(self.pBname)

    def __findBatchStats(self):
        #Finds the statistics of all of the profiles in all of the multisims
        self.lineStats = []
        print("Fitting Profiles...")
        bar = pb.ProgressBar(len(self.profiless))
        bar.display()
        for profiles in self.profiless:
            self.lineStats.append(self.__findSimStats(profiles))
            bar.increment()
            bar.display()
        bar.display(force = True)
        self.__findSampleStats()

    def __findSimStats(self, profiles):
        #Finds the statistics of all profiles in a given list
        simStats = []
        for profile in profiles:
            simStats.append(self.__findProfileStats(profile))
        return simStats

    def __findProfileStats(self, profile):
        if self.usePsf:
            try:
                profile = self.con.convolve(profile, self.psf, boundary='extend')
            except: print("No point spread function available")

        if self.statType.lower() == 'moment':
            #Finds the moment statistics of a single line profile
            maxMoment = 5
            moment = np.zeros(maxMoment)
            for mm in np.arange(maxMoment):
                    moment[mm] = np.dot(profile, self.env.lamAx**mm)

            power = moment[0] 
            mu = moment[1] / moment[0]
            sigma = np.sqrt(moment[2]/moment[0] - (moment[1]/moment[0])**2)
            skew = (moment[3]/moment[0] - 3*mu*sigma**2 - mu**3) / sigma**3
            kurt = (moment[4] / moment[0] - 4*mu*moment[3]/moment[0]+6*mu**2*sigma**2 + 3*mu**4) / sigma**4 - 3
        else:
            #Fits a gaussian to a single line profile
            def gauss_function(x, a, x0, sigma):
                return a*np.exp(-(x-x0)**2/(2*sigma**2))

            psfSig = (self.env.psfSig if self.usePsf else 0)

            sig0 = np.sum((self.env.lamAx - self.env.lam0)**2)/len(profile)
            amp0 = np.max(profile)
            profNorm = profile - np.min(profile)
            popt, pcov = curve_fit(gauss_function, self.lamAx, profNorm, p0 = [amp0, self.env.lam0, sig0])
            
            power = popt[0] * np.sqrt(np.pi * 2 * popt[2]**2)
            mu = popt[1]

            psfSig = 1.030086*psfSig #This is the factor that makes it the same before and after psf

            sigma = np.sqrt(np.abs(popt[2])**2 - psfSig**2) #Subtract off the PSF width
            skew = 0
            kurt = 0
            #fit = gauss_function(self.env.lamAx, popt[0], mu, sigma)
            #plt.plot(self.env.lamAx, profile)
            #plt.plot(self.env.lamAx, fit)
            #plt.show()

        return [power, mu, sigma, skew, kurt]



    def makePSF(self, angSig):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import astropy.convolution as con
            self.con = con
        self.lamAx = self.env.makeLamAxis(self.env.Ln, self.env.lam0, self.env.lamPm)
        if angSig is not None:
            diff = np.abs(self.lamAx[1] - self.lamAx[0])
            pix = int(np.ceil(angSig/diff))
            self.psf = self.con.Gaussian1DKernel(pix)
        else: self.psf = None

    def __findSampleStats(self):
        #Finds the mean and varience of each of the statistics for each multisim
        self.stat =  [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
        self.statV = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
        self.allStd = []
        for vars, impact in zip(self.lineStats, self.labels):
            allAmp = [x[0] for x in vars]
            allMean = [x[1] for x in vars]
            allMeanC = [x[1] - self.env.lam0 for x in vars]
            allStd = [x[2] for x in vars]
            allSkew = [x[3] for x in vars]
            allKurt = [x[4] for x in vars]
            
            #Wavelength Units
            self.assignStat(0, allAmp)
            self.assignStat(1, allMeanC)
            self.assignStat(2, allStd)
            self.assignStat(3, allSkew)
            self.assignStat(4, allKurt)
            
            #Velocity Units
            self.assignStatV(0, allAmp)
            self.assignStatV(3, allSkew)
            self.assignStatV(4, allKurt)

            mean1= self.__mean2V(np.mean(allMean))
            std1 = np.std([self.__mean2V(x) for x in allMean])
            self.assignStatV2(1, mean1, std1)

            self.allStd.append([self.__std2V(x) for x in allStd])
            mean2= self.__std2V(np.mean(allStd))
            std2 = self.__std2V(np.std(allStd))
            self.assignStatV2(2, mean2, std2)

    def assignStat(self, n, var):
        self.stat[n][0].append(np.mean(var))
        self.stat[n][1].append(np.std(var))

    def assignStatV(self, n, var):
        self.statV[n][0].append(np.mean(var))
        self.statV[n][1].append(np.std(var))

    def assignStatV2(self, n, mean, std):
        self.statV[n][0].append(mean)
        self.statV[n][1].append(std)

    def __mean2V(self, mean):
        #Finds the redshift velocity of a wavelength shifted from lam0
        return self.env.cm2km((self.env.ang2cm(mean) - self.env.ang2cm(self.env.lam0)) * self.env.c / 
                (self.env.ang2cm(self.env.lam0)))

    def __std2V(self, std):
        return np.sqrt(2) * self.env.cm2km(self.env.c) * (std / self.env.lam0)

    def plotProfiles(self, max):
        if max is not None:
            for profiles, impact in zip(self.profiles, self.doneLabels):
                plt.figure()
                plt.title('Impact: ' + str(impact))
                plt.xlabel('Wavelength')
                count = 0
                for profile in profiles:
                    if count < max:
                        plt.plot(self.env.lamAx, profile)
                        count += 1
                plt.show()

    def plotProfTogether(self, average = False, norm = False, log = False):
        plt.figure()
        plt.title('Profiles vs Impact')
        plt.xlabel('Wavelength')
        plt.ylabel('Intensity')

        for profiles, impact in zip(self.profiless, self.doneLabels):
            profsum = np.zeros_like(profiles[0])
            count = 0
            for profile in profiles:
                if norm: profile /= np.amax(profile)
                if average:
                    profsum += profile
                    count += 1
                else:
                    plt.plot(self.env.lamAx, profile, label = impact)
                    break
            if average:
                profsum /= count
                plt.plot(self.env.lamAx, profile, label = impact)
        if log: plt.yscale('log')
        plt.legend()
        plt.show()
                  
    def plotStats(self):
        f, axArray = plt.subplots(3, 1, sharex=True)
        mm = 0
        titles = ['amp', 'mean', 'sigma']
        ylabels = ['', 'Angstroms', 'Angstroms']
        for ax in axArray:
            if mm == 0: ax.set_yscale('log')
            ax.errorbar(self.labels, self.stat[mm][0], yerr = self.stat[mm][1], fmt = 'o')
            ax.set_title(titles[mm])
            ax.set_ylabel(ylabels[mm])
            mm += 1
            ax.autoscale(tight = False)
        ax.set_xlabel(self.xlabel)
        plt.show(False)

    #Main Plots
    def plotStatsV(self):
        f, axArray = plt.subplots(5, 1, sharex=True)
        f.canvas.set_window_title('Coronasim')
        doRms = True
        labels = self.getLabels()
        mm = 0
        titles = ['Intensity', 'Mean Redshift', 'Line Width', 'Skew', 'Excess Kurtosis']
        ylabels = ['', 'km/s', 'km/s', '', '']
        #import copy
        #thisBlist = copy.deepcopy(self.Blist)
        try:
            self.completeTime
        except: self.completeTime = 'Incomplete Job'
        f.suptitle(str(self.batchName) + ': ' + str(self.completeTime) + '\n Wavelength: ' + str(self.env.lam0) + 
            ' Angstroms\nLines per Impact: ' + str(self.Npt) + '\n Envs: ' + str(self.Nenv) + 
            '; Lines per Env: ' + str(self.Nrot) + '\n                                                                      statType = ' + str(self.statType))
        for ax in axArray:
            if mm == 0: ax.set_yscale('log') #Set first plot to log axis
            ax.errorbar(labels, self.statV[mm][0], yerr = self.statV[mm][1], fmt = 'o')
            if mm == 2 and doRms: #Plot Vrms
                ax.plot(labels, self.thisV) 

                #Put numbers on plot of widths
                for xy in zip(labels, self.statV[mm][0]): 
                    ax.annotate('(%.2f)' % float(xy[1]), xy=xy, textcoords='data')
            if mm == 1 or mm == 3 or mm == 4: #Plot a zero line
                ax.plot(labels, np.zeros_like(labels))
            ax.set_title(titles[mm])
            ax.set_ylabel(ylabels[mm])
            mm += 1
            spread = 0.05
            ax.set_xlim([labels[0]-spread, labels[-1]+spread]) #Get away from the edges
        ax.set_xlabel(self.xlabel)
        grid.maximizePlot()
        plt.show()

    def getLabels(self):
        try:
            labels = np.asarray(self.doneLabels)
        except: 
            labels = np.arange(len(self.profiles))
            doRms = False
        return labels

    def plotWidth(self):
        f = plt.figure()
        f.canvas.set_window_title('Coronasim')
        doRms = True
        try:
            labels = np.asarray(self.doneLabels)
        except: 
            labels = np.arange(len(self.profiles))
            doRms = False
        try:
            self.completeTime
        except: self.completeTime = 'Incomplete Job'
        f.suptitle(str(self.batchName) + ': ' + str(self.completeTime) + '\n Wavelength: ' + str(self.env.lam0) + 
            ' Angstroms\nLines per Impact: ' + str(self.Npt) + ';     Envs: ' + str(self.Nenv) + 
            '; Lines per Env: ' + str(self.Nrot) + '\n  usePsf = '+ str(self.usePsf) + '                                          statType = ' + str(self.statType))

        str1 = "{}: {}\nWavelength: {} Angstroms\nEnvs: {}; Lines per Env: {}; Lines per Impact: {}\n".format(self.batchName, self.completeTime, self.env.lam0, self.Nenv, self.Nrot, self.Npt)
        str2 = "usePsf: {}                                          statType: {}".format(self.usePsf, self.statType)
        f.suptitle(str1 + str2)

        #Display the run flags
        height = 0.9
        left = 0.18
        shift = 0.1

        plt.figtext(left, height + 0.04, "Wind: {}".format(self.copyPoint.useWind))
        plt.figtext(left, height + 0.02, "Waves: {}".format(self.copyPoint.useWaves))
        plt.figtext(left, height, "B: {}".format(self.copyPoint.bWasOn))

        #Plot the actual distribution of line widths in the background
        self.histlabels = []
        edges = []
        hists = []
        low = []
        self.medians = []
        high = []

        small = 1e16
        big = 0
        for stdlist in self.allStd:
            newBig = np.abs(np.ceil(np.amax(stdlist)))
            newSmall = np.abs(np.floor(np.amin(stdlist)))
            
            small = int(min(small, newSmall))
            big = int(max(big, newBig))
        throw = int(np.abs(np.ceil(big-small)))

        for stdlist, label in zip(self.allStd, self.doneLabels):

            hist, edge = np.histogram(stdlist, throw, range = [small,big])
            hist = hist / np.amax(hist)
            self.histlabels.append(label)
            edges.append(edge)
            hists.append(hist)

            
            quarts = np.percentile(stdlist, self.qcuts)
            low.append(quarts[0])
            self.medians.append(quarts[1])
            high.append(quarts[2])


        array = np.asarray(hists).T
        diff = np.average(np.diff(np.asarray(self.histlabels)))
        self.histlabels.append(self.histlabels[-1] + diff)
        ed = edges[0][:-1]
        xx, yy = np.meshgrid(self.histlabels-diff/2, ed)
        self.histlabels.pop()

        plt.pcolormesh(xx, yy, array, cmap = 'YlOrRd', label = "Sim Hist")
        cbar = plt.colorbar()
        cbar.set_label('Number of Lines')

        #Plot the confidence intervals
        plt.plot(self.histlabels, low, 'c:', label = "{}%".format(self.qcuts[0]), drawstyle = "steps-mid")
        plt.plot(self.histlabels, self.medians, 'c--', label = "{}%".format(self.qcuts[1]), drawstyle = "steps-mid")
        plt.plot(self.histlabels, high, 'c:', label = "{}%".format(self.qcuts[2]), drawstyle = "steps-mid")

        #Plot the Statistics of the Lines
        self.lineWidths = self.statV[2][0]
        self.lineWidthErrors = self.statV[2][1]
        plt.errorbar(self.histlabels, self.lineWidths, yerr = self.lineWidthErrors, fmt = 'bo', label = 'Simulation')

        #Do the chi-squared test
        self.chiTest()
        height = 0.9
        left = 0.65
        shift = 0.1
        plt.figtext(left, height + 0.04, "Fit to the Median")
        plt.figtext(left, height + 0.02, "Chi2 = {:0.3f}".format(self.chi_mid))
        plt.figtext(left, height, "Chi2_R = {:0.3f}".format(self.rChi_mid))
        plt.figtext(left + shift, height + 0.04, "Fit to the Mean")
        plt.figtext(left + shift, height + 0.02, "Chi2 = {:0.3f}".format(self.chi_mean))
        plt.figtext(left + shift, height, "Chi2_R = {:0.3f}".format(self.rChi_mean))

        #Plot the Hahn Measurements
        try:
            plt.errorbar(self.env.hahnAbs, self.env.hahnPoints, yerr = self.env.hahnError, fmt = 'gs', label = 'Hahn Observations')
        except: print("Failed to plot hahn measurements")

        #Plot the expected values
        try:
            plt.plot(self.doneLabels, self.hahnV, label = "HahnV", color = 'g')        
            
            plt.plot(self.doneLabels, self.thisV, label = 'Expected', color = 'b') 
        except: print("Failed to plot fit lines")

        #Put numbers on plot of widths
        for xy in zip(self.histlabels, self.statV[2][0]): 
            plt.annotate('(%.2f)' % float(xy[1]), xy=xy, textcoords='data')

        plt.title('Line Width')
        plt.legend(loc =2)
        plt.ylabel('Km/s')
        spread = 0.02
        plt.xlim([self.histlabels[0]-spread, self.histlabels[-1]+spread]) #Get away from the edges
        plt.xlabel(self.xlabel)
        grid.maximizePlot()
        plt.show()

    def chiTest(self):
        #import pdb 
        #pdb.set_trace()
        #self.doneLabels
        #self.lineWidths
        #self.lineWidthErrors
        #self.medians
        #self.env.hahnAbs
        #self.env.hahnPoints
                
        self.hahnMids = np.interp(self.env.hahnAbs, self.histlabels, self.medians)
        self.hahnMeans = np.interp(self.env.hahnAbs, self.histlabels, self.lineWidths)
        self.hahnMidErrors = np.interp(self.env.hahnAbs, self.histlabels, self.lineWidthErrors)

        self.chi_mean = 0
        self.chi_mid = 0
        N = 0
        for hWidth, hError, miWidth, meWidth, mError in zip(self.env.hahnPoints, self.env.hahnError, self.hahnMids, self.hahnMeans, self.hahnMidErrors):
            N += 1
            self.chi_mid += (hWidth - miWidth)**2 / (hError * mError)
            self.chi_mean += (hWidth - meWidth)**2 / (hError * mError)
        
        self.rChi_mid = self.chi_mid / N
        self.rChi_mean = self.chi_mean / N

        

    def plot(self, width):
        if width:
            self.plotWidth()
        else:
            self.plotStatsV()

    def save(self, batchName = None, keep = False, printout = False, dumpEnvs = False):

        if batchName is None: batchName = self.batchName
    
        self.slash = os.path.sep

        #Save with all data
        if keep:
            batchPath = '..' + self.slash + 'dat' + self.slash + 'batches' + self.slash + batchName + '_keep.batch'            
            script_dir = os.path.dirname(os.path.abspath(__file__))   
            absPath = os.path.join(script_dir, batchPath)  
            with open(absPath, 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        
        #Save Without Data
        
        batchPath = '..' + self.slash + 'dat' + self.slash + 'batches' + self.slash + batchName + '.batch'

        sims = self.sims
        self.sims = []
        if dumpEnvs:
            envs = self.envs
            self.envs = []

        script_dir = os.path.dirname(os.path.abspath(__file__))   
        absPath = os.path.join(script_dir, batchPath)  
        with open(absPath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        self.sims = sims
        if dumpEnvs:
            self.envs = envs
        #self.env = env
        if printout: print('\nFile Saved')
            
    def show(self):
        #Print all properties and values
        myVars = vars(self)
        print("\nBatch Properties\n")
        for ii in sorted(myVars.keys()):
            if not "stat" in ii.lower():
                print(ii, " : ", myVars[ii])
                
    def showAll(self):
        #Print all properties and values
        myVars = vars(self)
        print("\nBatch Properties\n")
        for ii in sorted(myVars.keys()):
            print(ii, " : ", myVars[ii])

    def doOnePB(self):
        pBs = np.asarray(self.pBss[-1])
        self.pBavg = np.average(pBs)
        self.pBstd = np.std(pBs)

    def calcFfiles(self):
        #Calculate the f parameters 
        #Make sure to have the B field and waves and wind on if you want this to work
        print("\nCalculating f Files:")
        sys.stdout.flush()
        name = self.fName
        folder = "../dat/data/"
        file1 = os.path.normpath(folder + 'f1_' + name + '.txt')
        file2 = os.path.normpath(folder + 'f2_' + name + '.txt')
        file3 = os.path.normpath(folder + 'f3_' + name + '.txt')
        f1 = []
        f2 = []
        f3 = []
        absic = []

        with open(file1, 'w') as f1out:
            with open(file2, 'w') as f2out:
                with open(file3, 'w') as f3out:

                    for urProjs, rmsProjs, temProjs, b in zip(self.urProjss, self.rmsProjss, self.temProjss, self.doneLabels):
                        #Simulate one line across the top of the sun
                        #lineSim = simulate(grd, env, N = rez, findT = True, getProf = True)

                        urProj = np.average(urProjs)
                        rmsProj = np.average(rmsProjs)
                        temProj = np.average(temProjs)
                        #print(rmsProj)
                        #sys.stdout.flush()
                        #Simulate a point directly over the pole and average
                        ptUr = []
                        ptRms = []
                        ptTem = []
                        for env in self.envs:
                            point = simpoint([0,0,b], grid = grid.defGrid().impLine, env = env)
                            ptUr.append(point.ur)
                            ptRms.append(point.vRms)
                            ptTem.append(point.T)
                        #print(ptRms)
                        #sys.stdout.flush()
                        pUr = np.average(ptUr)
                        pRms = np.average(ptRms)
                        pTem = np.average(ptTem)
                        #print(pRms)
                        #sys.stdout.flush()
                        urProjA = urProj/pUr
                        rmsProjA = rmsProj/pRms
                        temProjA = temProj/pTem

                        #Store the info
                        print(str(b) + ' ur: ' + str(urProjA)+ ', rms: ' + str(rmsProjA)+ ', tem: ' + str(temProjA))
                        sys.stdout.flush()
                        f1.append(urProjA)
                        f2.append(rmsProjA)
                        f3.append(temProjA)
                        absic.append(b)
                        f1out.write('{}   {}\n'.format(b,urProjA))
                        f1out.flush()
                        f2out.write('{}   {}\n'.format(b,rmsProjA))
                        f2out.flush()
                        f3out.write('{}   {}\n'.format(b,temProjA))
                        f3out.flush()
                        #lineSim.plot('rmsProj')
                    for env in self.envs:
                        env._fLoad(name)
                    plt.plot(np.asarray(absic), np.asarray(f1), label = 'f1')
                    plt.plot(np.asarray(absic), np.asarray(f2), label = 'f2')
                    plt.plot(np.asarray(absic), np.asarray(f3), label = 'f3')
                    plt.legend()
                    plt.axhline(1, color = 'k')
                    plt.show()

# For doing a multisim at many impact parameters
class impactsim(batchjob):
    def __init__(self, batchName, envs, Nb = 10, iter = 1, b0 = 1.05, b1= 1.50, N = (1500, 10000), 
            rez = None, size = None, timeAx = [0], length = 10, printSim = False, printOut = True, printMulti = True, fName = None, qcuts = [16,50,84], spacing = 'lin'):
        comm = MPI.COMM_WORLD
        self.size = comm.Get_size()
        self.root = comm.Get_rank() == 0
        self.count = 0
        self.fName = fName
        self.Nb = Nb
        self.batchName = batchName
        self.timeAx = timeAx
        self.qcuts = qcuts 
        try: self.Nenv = len(envs)
        except: self.Nenv = 1

        #Lines per environment
        if self.root and self.size < self.Nenv:
            print('**Warning: More envs than PEs, will take additional time**')
        self.Nrot = np.floor(iter * max(1, self.size / self.Nenv))

        #Lines per impact
        self.Npt = self.Nrot * self.Nenv 
        #print("{} lines per run".format(self.Npt))
        #Total Lines
        self.Ntot = self.Npt * self.Nb
        #if self.root: 
        #    import pdb 
        #    pdb.set_trace()
        #self.comm.barrier()
        self.print = printOut
        self.printMulti = printMulti
        self.printSim = printSim

        self.N = N

        base = 1000

        #if b1 is not None: 
        if b1 is not None: 
            if spacing.casefold() == 'log'.casefold():
                steps = np.linspace(b0,b1,Nb)
                logsteps = base**steps
                logsteps = logsteps - np.amin(logsteps)
                logsteps = logsteps / np.amax(logsteps) * (b1-b0) + b0
                self.labels = logsteps

                #self.labels = np.round(np.logspace(np.log(b0)/np.log(base),np.log(b1)/np.log(base), Nb, base = base), 4)
            else: self.labels = np.round(np.linspace(b0,b1,Nb), 4)
        else: self.labels = np.round([b0], 4)
        
        self.impacts = self.labels
        self.xlabel = 'Impact Parameter'    
        self.fullBatch = []
        for ind in self.impacts:
            self.fullBatch.append(grid.rotLines(N = self.Nrot, b = ind, rez = rez, size = size, x0 = length)) 

        super().__init__(envs)
        
        return
        
    def makeVrms(self):
            #self.count += 1
            #print("The count is {}".format(self.count))
            #sys.stdout.flush()
        self.thisV = []
        self.hahnV = []
        #if self.root: 
        #    import pdb 
        #    pdb.set_trace()
        if super().lockFlags:
            try: copyPoint = self.copyPoint
            except: copyPoint = None
        else: copyPoint = None

        for impact in self.doneLabels:
            ptUr = []
            ptRms = []
            ptTem = []
            for env in self.envs:
                point = simpoint([0,0,impact], grid = grid.defGrid().impLine, env = env, copyPoint = copyPoint)
                ptUr.append(point.ur)
                ptRms.append(point.vRms)
                ptTem.append(point.T)

            pUr = np.average(ptUr)
            pRms = np.average(ptRms)
            pTem = np.average(ptTem)

            vTh = 2 * self.env.KB * pTem / self.env.mI

            wind =   (self.env.interp_f1(impact) * pUr)**2 
            rms =    (self.env.interp_f2(impact) * pRms)**2
            thermal = (self.env.interp_f3(impact) * vTh)**1
            V = np.sqrt( (thermal + wind + rms) )

            self.thisV.append(self.env.cm2km(V))
            self.hahnV.append(self.hahnFit(impact))

    def hahnFit(self, r, r0 = 1.05, vth = 25.8, vnt = 32.2, H = 0.0657):
        veff = np.sqrt(vth**2+vnt**2 * np.exp(-(r-r0)/(r*r0*H))**(-1/2))
        return veff


def pbRefinement(envsName, params, MIN, MAX, tol):
    #This function compares the pB at a height of zz with and without B, and finds the Bmin which minimizes the difference
    def runAtBminFull(params, Bmin):
        #This runs a multisim at a particular Bmin and returns the pB
        comm = MPI.COMM_WORLD
        root = comm.Get_rank() == 0
        simpoint.useB = True
        simpoint.Bmin = Bmin
        
        bat = impactsim(*params)
        if root:
            pBnew = bat.pBavg
            print("B = {}, pB = {}".format(Bmin, pBnew))
            sys.stdout.flush()
        else:
            pBnew = np.empty(1)
        comm.Bcast(pBnew, root=0)
        return pBnew
                 
    def bisection(a,b,tol,f,pBgoal):
        #Given a function f, a tolerance, and a goal, this returns the value that solves for the goal.
        comm = MPI.COMM_WORLD
        root = comm.Get_rank() == 0
        Nmax = 20
        N = 0
        c = (a+b)/2.0
        fc = f(c) - pBgoal
        flist = {}
        flist[c] = fc

        while np.abs(fc) > tol:   
            N += 1
            if N > Nmax:
                if root: print("I failed to converge")
                break

            try: fc = flist[c]
            except: 
                fc = f(c) - pBgoal
                flist[c] = fc
            try: fa = flist[a]
            except: 
                fa = f(a) - pBgoal
                flist[a] = fa

            if fc == 0:
                return c
            elif fa*fc < 0:
                b = c
            else:
                a = c
            c = (a+b)/2.0
        if root: 
            print("Converged to {}, with pB = {}, in {} iterations".format(c,fc+pBgoal, N))
            print("The goal was {}".format(pBgoal))
        return c
   
    comm = MPI.COMM_WORLD
    root = comm.Get_rank() == 0
    if root: 
        print("Beginning Refinement...")
        sys.stdout.flush()

    zz = params[4]
    len = params[10]
    rez = params[7]          
    refenv = params[1][0]
    refgrd = grid.sightline([-len,1e-8,zz], [len,1e-8,zz], findT = True)

    from functools import partial
    runAtBmin = partial(runAtBminFull, params)
            
    #Get the reference line pB
    simpoint.useB = False
    lineSim = simulate(refgrd, refenv, N = rez, findT = True, getProf = True)
    pBgoal = lineSim.pB
    if root: 
        print("The goal is pB = {}".format(pBgoal))
        sys.stdout.flush()

    #Converge to that line        
    return bisection(MIN,MAX,tol,runAtBmin,pBgoal)
    

     



    #def runAtImpact(params, b):
    #    #This runs a multisim at a particular Bmin and returns the pB
    #    comm = MPI.COMM_WORLD
    #    root = comm.Get_rank() == 0
    #    params = ["fCalcs", envs, impactPoints, iterations, b, None, N_line, rez, size, timeAx, length, False, False, False]
    #    bat = impactsim(batchName, envs, impactPoints, iterations, b0, b1, N_line, rez, size, timeAx, length, printSim)
    #    if root:
    #        pBnew = bat.pBavg
    #        print("B = {}, pB = {}".format(Bmin, pBnew))
    #        sys.stdout.flush()
    #    else:
    #        pBnew = np.empty(1)
    #    comm.Bcast(pBnew, root=0)
    #    return pBnew
#def plotpB(maxN = 100):
#        path = os.path.normpath("../dat/pB/*.txt")
#        files = glob.glob(path)

#        ind = 0
#        for file in files:
#            if ind < maxN: 
#                x = np.loadtxt(file)
#                absiss = x[:,0]
#                pBavg = x[:,1]
#                pBstd = x[:,2]
#                label = file.rsplit(os.path.sep, 1)[-1]
#                plt.plot(absiss, pBavg, '-o', label = label)#, yerr = pBstd) 
#            ind += 1

#        plt.legend()
#        plt.yscale('log')
#        plt.ylabel('pB')
#        plt.xlabel('Impact Parameter')
#        plt.show()        
        
        
    #def doPB(self, filename):
    #    if filename is not None:
    #        self.pBavg = []
    #        self.pBstd = []
    #        path = os.path.normpath("../dat/pB/" + filename + ".txt")
    #        with open(path, 'w') as f:
    #            for label, pBs in zip(self.getLabels(),self.pBs):
    #                data = np.asarray(pBs)
    #                avg = np.average(data)
    #                std = np.std(data)

    #                self.pBavg.append(avg)
    #                self.pBstd.append(std)

    #                f.write("{}    {}    {}\n".format(label, avg, std))
    #                f.flush()

                #plt.errorbar(self.getLabels(), pBavg, yerr = pBstd, fmt = 'o')
                #plt.yscale('log')
                ##plt.semilogy(self.getLabels(), pB)
                #plt.show()        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

def nothing():
    pass

