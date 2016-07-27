# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:13:05 2016


@author: chgi7364
"""

#print('Loading Dependencies...')
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from scipy import io
from scipy import ndimage
from scipy import interpolate as interp
from scipy.stats import norm
import scipy.stats as stats
from scipy.optimize import curve_fit

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
##                          Environment                                                                                                    ##
####################################################################

#Environment Class contains simulation parameters
class environment:

    #Environment Class contains simulation parameters
    slash = os.path.sep
    datFolder = '..' + slash + 'dat' + slash

    rel_def_Bfile = datFolder + 'mgram_iseed0033.sav'    
    def_Bfile = absPath(rel_def_Bfile)

    rel_def_xiFile1 = datFolder + 'xi_1.dat'    
    def_xiFile1 = absPath(rel_def_xiFile1)

    rel_def_xiFile2 = datFolder + 'xi_2.dat'    
    def_xiFile2 = absPath(rel_def_xiFile2)

    rel_def_bkFile = datFolder + 'plasma_background.dat'    
    def_bkFile =absPath(rel_def_bkFile)
    
    #Parameters
    rstar = 1
    B_thresh = 6.0
    fmax = 8.2
    theta0 = 28
    S0 = 7.0e5

    r_Mm = 695.5 #Rsun in Mega meters

    #Constants
    c = 2.998e10 #cm/second (base velocity unit is cm/s)
    KB = 1.380e-16 #ergs/Kelvin

    #For randomizing wave angles/init-times
    primeSeed = 27
    randOffset = 0  

    #Element Being Observed
    #mi = 1.6726219e-24 #grams per hydrogen
    mi = 9.2732796e-23 #grams per Iron

    #LamAxis Stuff
    Ln = 100
    lam0 = 200
    lamPm = 0.5


    streamRand = np.random.RandomState()
    primeRand = np.random.RandomState(primeSeed)

    def __init__(self, Bfile = None, bkFile = None):
    
        print('  Loading Environment...', end = '', flush = True)

        self.makeLamAxis(self.Ln, self.lam0, self.lamPm)

        #Load Bmap
        if Bfile is None: self.Bfile = self.def_Bfile 
        else: self.Bfile = self.__relPath(Bfile)
        self.__loadBMap()
        self.__labelStreamers()
        self.analyze_BMap()

        #Load Xi
        self.xiFile1 = self.def_xiFile1 
        self.xiFile2 = self.def_xiFile2 

        x = np.loadtxt(self.xiFile1, skiprows=1)
        self.xi1_t = x[:,0]
        self.xi1_raw = x[:,1]
        self.last_xi1_t = x[-1,0]

        y = np.loadtxt(self.xiFile2, skiprows=1)
        self.xi2_t = y[:,0]
        self.xi2_raw = y[:,1]
        self.last_xi2_t = y[-1,0]

        #Load Plasma Background
        if bkFile is None: self.bkFile = self.def_bkFile 
        else: self.bkFile = self.__relPath(bkFile)
        x = np.loadtxt(self.bkFile, skiprows=10)
        self.bk_dat = x
        self.rx_raw = x[:,0]
        self.rho_raw = x[:,1]
        self.ur_raw = x[:,2]
        self.vAlf_raw = x[:,3]
        self.T_raw = x[:,4]

        print("Done")
        print('')

    def smallify(self):
        self.label_im = []

    def save(self, name):
        with open(self.__relPath(name), 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        
    def randomize(self):
        self.randOffset = int(self.primeRand.uniform(0, 10000))

    def setOffset(self, offset):
        self.randOffset = offset

    def __relPath(self, path):
        #Converts a relative path to an absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))   
        rel = os.path.join(script_dir, path)    
        return rel

    def __find_nearest(self,array,value):
        #Returns the index of the point most similar to a given value
        idx = (np.abs(array-value)).argmin()
        return idx

    def interp_rx_dat(self, rx, array):
        #Interpolates an array(rx)
        if rx < 1. : return math.nan
        rxInd = int(self.__find_nearest(self.rx_raw, rx))
        val1 = array[rxInd]
        val2 = array[rxInd+1]
        slope = val2 - val1
        step = self.rx_raw[rxInd+1] - self.rx_raw[rxInd]
        discreteRx = self.rx_raw[rxInd]
        diff = rx - discreteRx
        diffstep = diff / step
        return val1 + diffstep*(slope)

    def cm2km(self, var):
        return var * 1e-5

    def km2cm(self, var):
        return var * 1e5

    def ang2cm(self, var):
        return var * 1e-8

    def cm2ang(self, var):
        return var * 1e8

    ## Velocities #####################################################################

    def findVrms(self, rx, B):
        #RMS Velocity
        densfac = self.__findDensFac(B)
        ur = self.__findUr(rx, densfac)
        vAlf = self.__findAlf(rx,densfac)
        vPh = ur + vAlf
        rho = self.__findRho(rx, densfac)
        Hfit  = 2.2*(self.fmax/10.)**0.62
        f = self.fmax + (1.-self.fmax)*np.exp(-((rx-1.)/Hfit)**1.1)
        return np.sqrt(self.S0*vAlf/((vPh)**2*rx**2*f*rho))

    def __findDensFac(self, B):
        # Find the density factor
        if np.abs(B) < np.abs(self.B_thresh):
            return 1
        else:
            return (np.abs(B) / self.B_thresh)**0.5

    def __findUr(self, rx, densfac):
        #Wind Velocity
        return self.interp_rx_dat(rx, self.ur_raw) / densfac

    def __findAlf(self, rx, densfac):
        #Alfen Velocity
        return self.interp_rx_dat(rx, self.vAlf_raw) / np.sqrt(densfac)

    def __findRho(self, rx, densfac):
        return self.interp_rx_dat(rx, self.rho_raw) * densfac 


  ## Magnets ##########################################################################
        
    def __loadBMap(self):
        Bobj = io.readsav(self.Bfile)
        self.BMap_raw = Bobj.get('data_cap')
        #plt.imshow((np.abs(simpoint.BMap_raw)), cmap = "winter_r")
        #plt.colorbar()
        #plt.show()
        self.BMap_x = Bobj.get('x_cap')
        self.BMap_y = Bobj.get('y_cap')
        self.BMap = interp.RectBivariateSpline(self.BMap_x, self.BMap_y, self.BMap_raw)

    def __labelStreamers(self, thresh = 0.9):
        #Take magnitude and find valid region
        bdata = np.abs(self.BMap_raw)
        validMask = bdata != 0

        #Find all above the threshold and label
        blist = bdata.flatten().tolist()
        bmean =  np.mean([v for v in blist if v!=0])
        bmask = bdata > bmean * thresh
        label_im, nb_labels = ndimage.label(bmask)
        
        #Create seeds for voronoi
        coord = ndimage.maximum_position(bdata, label_im, np.arange(1, nb_labels))
        coordinates = []
        for co in coord: coordinates.append(co[::-1])
        
        #Get voronoi background
        self.label_im, self.nb_labels = self.__voronoify_sklearn(label_im, coordinates)
        
        #Add in threshold regions
        highLabelIm = label_im + self.nb_labels
        self.label_im *= np.logical_not(bmask)
        self.label_im += highLabelIm * bmask
        
        #Clean Edges
        self.label_im *= validMask
        
        if False: #Plot Slice of Map
            (label_im + 40) * validMask
            plt.imshow(self.label_im, cmap = 'Paired', interpolation='none')
            plt.imshow(highLabelIm, cmap = 'Set1', interpolation='none', alpha = 0.5)
            plt.xlim(600,950)
            plt.ylim(350,650)
            #for co in coordinates:
            #    plt.scatter(*co)
            plt.show()
        return

    def __voronoify_sklearn(self, I, seeds):
        import sklearn.neighbors as nb
        #Uses the voronoi algorithm to assign stream labels
        tree_sklearn = nb.KDTree(seeds)
        pixels = ([(r,c) for r in range(I.shape[0]) for c in range(I.shape[1])])
        d, pos = tree_sklearn.query(pixels)
        cells = defaultdict(list)

        for i in range(len(pos)):
            cells[pos[i][0]].append(pixels[i])

        I2 = I.copy()
        label = 0
        for idx in cells.values():
            idx = np.array(idx)
            label += 1
            #mean_col = I[idx[:,0], idx[:,1]].mode(axis=0) #The pretty pictures part
            I2[idx[:,0], idx[:,1]] = label
        return I2, label

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
        plt.plot(radius_Mm, hist2)
        plt.title('Distribution of Region Sizes')
        plt.xlabel('Radius (Mm)')
        plt.ylabel('Number of Regions')
        plt.show()


  ## Light ############################################################################

    def makeLamAxis(self, Ln = 100, lam0 = 200, lamPm = 0.5):
        self.lam0 = lam0
        self.lamAx = np.linspace(lam0 - lamPm, lam0 + lamPm, Ln)

        
#envs Class handles creation, saving, and loading of environments
class envs:

    #envs Class handles creation, saving, and loading of environments
    
    def __init__(self, envFolder = ''):
    
        self.slash = os.path.sep            
        self.dirPath = '..' + self.slash + 'dat' + self.slash + envFolder
        
        return
    
    def __loadEnv(self, path): 
        with open(path, 'rb') as input:
            return pickle.load(input)

    def __createEnvs(self, maxN = 1e8):
        files = glob.glob(absPath(self.dirPath + self.slash + '*.sav'))
        envs = []
        ind = 0
        for file in files:
            if ind < maxN: envs.append(environment(Bfile = file))
            ind += 1
        return envs
 
    def __saveEnvs(self, envs, maxN = 1e8, name = 'environment'):
        ind = 0
        for env in envs:
            if ind < maxN: env.save(absPath(self.dirPath + self.slash + name +'_' + str(ind) + '.env'))
            ind += 1
    
    def loadEnvs(self, maxN = 1e8):
        files = glob.glob(absPath(self.dirPath + self.slash + '*.env'))
        envs = []
        ind = 0
        for file in files:
            if ind < maxN: envs.append(self.__loadEnv(file))
            ind += 1
        return envs
        
    def processEnvs(self, maxN = 1e8, name = 'environment'):
        envs = self.__createEnvs(maxN)
        self.__saveEnvs(envs, maxN, name)
        return envs
            
       
####################################################################                            
##                           Simulation                                                                                                      ##
####################################################################

## Level 0: Simulates physical properties at a given coordinate
class simpoint:
    #Level 0: Simulates physical properties at a given coordinate
    def __init__(self, cPos = [0.1,0.1,1.5], grid = None, env = None, findT = True, pbar = None):
        #Inputs
        self.grid = grid
        self.env = env
        self.cPos = cPos
        self.pPos = self.cart2sph(self.cPos)
        self.rx = self.r2rx(self.pPos[0])
        
        if findT is None:
            self.findT = self.grid.findT
        else: self.findT = findT


        #Initialization
        self.findTemp()
        self.findFootB()
        self.findDensity()
        self.findTwave()
        self.__streamInit()
        self.findSpeeds()
        self.findIntensity(self.env.lam0, self.env.lam0)
        if pbar is not None:
            pbar.increment()
            pbar.display()


  ## Temperature ######################################################################

    def findTemp(self):
        self.T = self.interp_rx_dat(self.env.T_raw)


  ## Magnets ##########################################################################
        
    def findFootB(self):
        #Find B
        self.__findfoot_Pos()
        self.footB = self.env.BMap(self.foot_cPos[0], self.foot_cPos[1])[0][0]

    def __findfoot_Pos(self):
        #Find the footpoint of the field line
        Hfit  = 2.2*(self.env.fmax/10.)**0.62
        self.f     = self.env.fmax + (1.-self.env.fmax)*np.exp(-((self.rx-1.)/Hfit)**1.1)
        theta0_edge = self.env.theta0 * np.pi/180.
        theta_edge  = np.arccos(1. - self.f + self.f*np.cos(theta0_edge))
        edge_frac   = theta_edge/theta0_edge 
        coLat = self.pPos[1] /edge_frac
        self.foot_pPos = [self.env.rstar+1e-2, coLat, self.pPos[2]]
        self.foot_cPos = self.sph2cart(self.foot_pPos)
        
    def __findStreamIndex(self):
        self.streamIndex = self.env.randOffset + self.env.label_im[self.__find_nearest(self.env.BMap_x, self.foot_cPos[0])][
                                            self.__find_nearest(self.env.BMap_y, self.foot_cPos[1])]


  ## Density ##########################################################################
 
    def findDensity(self):
        #Find the densities of the grid point
        self.densfac = self.__findDensFac()
        self.rho = self.__findRho()

    def __findDensFac(self):
        # Find the density factor
        if np.abs(self.footB) < np.abs(self.env.B_thresh):
            return 1
        else:
            return (np.abs(self.footB) / self.env.B_thresh)**0.5

    def __findRho(self):
        return self.interp_rx_dat(self.env.rho_raw) * self.densfac  
   

  ## Velocity ##########################################################################

    def findSpeeds(self, t = 0):
        #Find all of the various velocities
        self.ur = self.__findUr()
        self.vAlf = self.__findAlf()
        self.vPh = self.__findVPh()
        self.vRms = self.__findvRms()
        self.findWaveSpeeds(t)

    def findWaveSpeeds(self, t = 0):
        #Find all of the wave velocities
        self.t1 = t - self.twave + self.alfT1
        self.t2 = t - self.twave + self.alfT2
        self.alfU1 = self.vRms*self.xi1(self.t1)
        self.alfU2 = self.vRms*self.xi2(self.t2)
        self.uTheta = self.alfU1 * np.sin(self.alfAngle) + self.alfU2 * np.cos(self.alfAngle)
        self.uPhi =   self.alfU1 * np.cos(self.alfAngle) - self.alfU2 * np.sin(self.alfAngle)
        self.pU = [self.ur, self.uTheta, self.uPhi]
        self.cU = self.__findCU() 
        self.vLOS = self.__findVLOS()      
       
    def __streamInit(self):
        self.__findStreamIndex()
        self.env.streamRand.seed(int(self.streamIndex))
        thisRand = self.env.streamRand.random_sample(3)
        self.alfT1 =  thisRand[0] * self.env.last_xi1_t
        self.alfT2 =  thisRand[1] * self.env.last_xi2_t
        self.alfAngle = thisRand[2] * 2 * np.pi

    def __findUr(self):
        #Wind Velocity
        #return (1.7798e13) * self.fmax / (self.num_den * self.rx * self.rx * self.f)
        return self.interp_rx_dat(self.env.ur_raw) / self.densfac

    def __findAlf(self):
        #Alfen Velocity
        #return 10.0 / (self.f * self.rx * self.rx * np.sqrt(4.*np.pi*self.rho))
        return self.interp_rx_dat(self.env.vAlf_raw) / np.sqrt(self.densfac)

    def __findVPh(self):
        #Phase Velocity
        return self.vAlf + self.ur 
    
    def __findvRms(self):
        #RMS Velocity
        S0 = 7.0e5 #* np.sqrt(1/2)
        return np.sqrt(S0*self.vAlf/((self.vPh)**2 * 
            self.rx**2*self.f*self.rho))/4 #TODO This factor of 4 is ad hoc

    def __findVLOS(self, nGrad = None):
        if nGrad is not None: self.nGrad = nGrad
        else: self.nGrad = self.grid.ngrad
        self.vLOS = np.dot(self.nGrad, self.cU)
        return self.vLOS

    def __findCU(self):
        #Finds the cartesian velocity components
        self.ux = -np.cos(self.pPos[2])*(self.ur*np.sin(self.pPos[1]) + self.uTheta*np.cos(self.pPos[1])) - np.sin(self.pPos[2])*self.uPhi
        self.uy = -np.sin(self.pPos[2])*(self.ur*np.sin(self.pPos[1]) + self.uTheta*np.cos(self.pPos[1])) - np.cos(self.pPos[2])*self.uPhi
        self.uz = self.ur*np.cos(self.pPos[1]) - self.uTheta*np.sin(self.pPos[1])
        return [self.ux, self.uy, self.uz]
    

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
        self.twave_rat = self.twave/self.twave_fit
        
    def setTime(self,t):
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


  ## Radiative Transfer ####################################################################

    def findIntensity(self, lam0 = 1000, lam = 1000):
        self.lam = lam #Angstroms
        self.lam0 = lam0 #Angstroms

        self.qt = 1

        self.lamLos =  self.vLOS * self.lam0 / self.env.c
        self.deltaLam = self.lam0 / self.env.c * np.sqrt(2 * self.env.KB * self.T / self.env.mi)
        self.lamPhi = 1/(self.deltaLam * np.sqrt(np.pi)) * np.exp(-((self.lam - self.lam0 - self.lamLos)/self.deltaLam)**2)
        self.intensity = self.rho**2 * self.qt * self.lamPhi * 1e33
        return self.intensity

        
  ## Misc Methods ##########################################################################

    def __find_nearest(self,array,value):
        #Returns the index of the point most similar to a given value
        idx = (np.abs(array-value)).argmin()
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
        val =  val1+ diffstep*(slope)
        return val

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
        
    def __relPath(self, path):
        #Converts a relative path to an absolute path
        script_dir = os.path.dirname(os.path.abspath(__file__))   
        rel = os.path.join(script_dir, path)    
        return rel
            
    def show(self):
        #Print all properties and values
        myVars = vars(self)
        print("\nSimpoint Properties")
        for ii in sorted(myVars.keys()):
            print(ii, " : ", myVars[ii])

    def Vars(self):
        return vars(self) 
#Inputs: (self, cPos = [0,0,1.5], grid = None, env = None, findT = None, pbar = None)


## Level 1: Initializes many Simpoints into a Simulation
class simulate: 
    #Level 1: Initializes many Simpoints into a Simulation
    def __init__(self, gridObj, envObj, N = None, iL = None, findT = None, printOut = False, nMin = None):
        self.print = printOut
        if self.print: print("Initializing Simulation...")
        self.grid  = gridObj
        self.findT = findT
        self.env = envObj
        self.profile = None
        self.adapt = False


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

    def simulate_now(self):
  
        if type(self.grid) is grid.plane: 
            doBar = True
            self.adapt = False
            self.Npoints = self.grid.Npoints
        else: doBar = False
        
        if doBar and self.print: bar = pb.ProgressBar(self.Npoints)
        self.sPoints = []
        self.steps = []
        self.pData = []
        if self.print: print("Beginning Simulation...")

        t = time.time()
        stepInd = 0
        rhoSum = 0
        tol = 0.5
        for cPos, step in self.grid: 

            thisPoint = simpoint(cPos, self.grid, self.env, self.findT) 

            if self.adapt:
                #Adaptive Mesh
                thisDens = thisPoint.intensity

                if (thisDens > tol) and self.grid.backflag:
                    self.grid.back()
                    self.grid.set2minStep()
                    self.grid.backflag = False
                    continue
                if thisDens <= tol:
                    self.grid.incStep(1.5)
                    self.grid.backflag = True

            stepInd += 1
            
            self.sPoints.append(thisPoint)
            self.steps.append(step)
            self.pData.append(thisPoint.Vars())
            
            if doBar and self.print: 
                bar.increment()
                bar.display()
        
        if doBar and self.print: bar.display(force = True)
        if self.print: print('Elapsed Time: ' + str(time.time() - t))

        self.Npoints = len(self.sPoints)
        if type(self.grid) is grid.sightline:
            self.shape = [self.Npoints, 1] 
        else: 
            self.shape = self.grid.shape
        self.shape2 = [self.shape[0], self.shape[1], -1]

    def get(self, myProperty, dim = None, scaling = 'None'):
        propp = np.array([x[myProperty] for x in self.pData])
        prop = propp.reshape(self.shape2)
        if not dim is None: prop = prop[:,:,dim]
        prop = prop.reshape(self.shape)
        if scaling.lower() == 'none':
            scaleProp = prop
        elif scaling.lower() == 'log':
            scaleProp = np.log10(prop)
        elif scaling.lower() == 'sqrt':
            scaleProp = np.sqrt(prop)
        else: 
            print('Bad Scaling - None Used')
            scaleProp = prop
        datSum = sum((v for v in scaleProp.ravel() if not math.isnan(v)))
        return scaleProp, datSum

    def plot(self, property, dim = None, scaling = 'None', cmap = 'jet'):
        scaleProp, datSum = self.get(property, dim, scaling)
        self.fig, ax = self.grid.plot(iL = self.iL)
 
        if type(self.grid) is grid.sightline:
            #Line Plot
            im = ax.plot(scaleProp)
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

        if dim is None:
            ax.set_title(property + ", scaling = " + scaling + ', sum = {}'.format(datSum))
        else:
            ax.set_title(property + ", dim = " + dim.__str__() + ", scaling = " + scaling + ', sum = {}'.format(datSum))

        grid.maximizePlot()
        plt.show(block = False)

    def compare(self, p1, p2, p1Scaling = 'None', p2Scaling = 'None', p1Dim = None, p2Dim = None):
        scaleprop = []
        scaleprop.append(self.get(p1, p1Dim, p1Scaling)[0])
        scaleprop.append(self.get(p2, p2Dim, p2Scaling)[0])
        fig, ax = self.grid.plot(iL = self.iL)
        fig.subplots_adjust(right=0.89)
        cbar_ax = fig.add_axes([0.91, 0.10, 0.03, 0.8], autoscaley_on = True)
      
        global cur_plot
        cur_plot = 0

        def plot1():
            im = ax.imshow(scaleprop[0], interpolation = 'none')
            ax.set_title(p1)
            fig.colorbar(im, cax=cbar_ax)
            fig.canvas.draw()

        def plot2():
            im = ax.imshow(scaleprop[1], interpolation = 'none')
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
        myVars = vars(self.sPoints[0])
        print("\nSimpoint Properties")
        for ii in sorted(myVars.keys()):
            print(ii, " : ", myVars[ii])

    def Vars(self):
        #Returns the vars of the simpoints
        return self.sPoints[0].Vars()

    def Keys(self):
        #Returns the keys of the simpoints
        return self.sPoints[0].Vars().keys()

    ####################################################################

    def getProfile(self):
        self.profile = self.lineProfile()
        return self.profile

    def plotProfile(self):
        if self.profile is None: self.lineProfile()
        plt.plot(self.env.lamAx, self.profile)
        plt.show()

    def setIntLam(self, lam):
        for point in self.sPoints:
            point.findIntensity(self.env.lam0, lam)
        
    def lineProfile(self):
        #Get a line profile at the current time
        profile = np.zeros_like(self.env.lamAx)
        index = 0
        for lam in self.env.lamAx:
            for point, step in zip(self.sPoints, self.steps):
                profile[index] += point.findIntensity(self.env.lam0, lam) * step
            index += 1
        return profile

## Time Dependence ######################################################
    def setTime(self, tt):
        for point in self.sPoints:
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
        intensity = np.zeros_like(self.sPoints)
        pInd = 0
        for point in self.sPoints:
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

        bar = pb.ProgressBar(self.Npoints*len(self.times))
        timeInd = 0
        for tt in self.times:
            for point in self.sPoints:
                point.setTime(tt)
                bar.increment()
                bar.display()   
            self.lineArray[timeInd][:] = self.lineProfile()
            timeInd += 1       
        bar.display(force = True)

        self.lineList = self.lineArray.tolist()

        self.plotLineArray_t()
        self.__fitGaussians_t()
        self.__findMoments_t()

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
        self.plotMomentStats_t()

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

        self.plotGaussStats_t()

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
#Inputs: (self, gridObj, envObj, N = None, iL = None, findT = None, printOut = False, nMin = None)


## Level 2: Initializes many simulations (MPI Enabled) for statistics
class multisim:
    #Level 2: Initializes many simulations
    def __init__(self, batch, envs, N = 1000, findT = None, printOut = False):
        self.print = printOut
        self.gridLabels = batch[1]
        self.oneBatch = batch[0]
        self.batch = []
        self.envInd = []
        
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
        self.MPI_init()
        #self.findProfiles()

    def MPI_init(self):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.root = self.rank == 0
        self.size = self.comm.Get_size()

        if self.root and self.print: 
            print('Running MultiSim: ' + time.asctime())
            t = time.time() #Print Stuff

        gridList = self.__seperate(self.batch, self.size)
        self.gridList = gridList[self.rank]      
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

        if self.root and self.print: 
            bar = pb.ProgressBar(len(self.gridList))
            bar.display()
        #self.simList = []
        profiles = []

        for grd, envInd in zip(self.gridList, self.envIndList):
            self.envs[envInd].randomize()
            t = time.time()
            simulation = simulate(grd, self.envs[envInd], self.N, findT = self.findT)
            #self.simList.append(simulation)
            profiles.append(simulation.getProfile())
            elapsed = time.time() - t
            if self.root and self.print:
                #print('')
                #print(elapsed)
                bar.increment()
                bar.display()
        if self.root and self.print: bar.display(force = True)
        self.profiles = self.comm.gather(profiles, root = 0)

    #def findProfiles(self):
    #    if self.root and self.print: 
    #        bar = pb.ProgressBar(len(self.gridList))
    #        print('Simulating Spectral Lines')
    #        bar.display()
    #    self.profiles = []
    #    for lsim in self.simList:
    #        self.profiles.append(lsim.getProfile())
    #        if self.root and self.print:
    #            bar.increment()
    #            bar.display()

    #    if self.root and self.print: bar.display(force = True)
    #    return self.profiles    

    def getLineArray(self):
        return np.asarray(self.profiles)
            
    def __seperate(self, list, N):
        #Breaks a list up into chunks
        import copy
        newList = copy.deepcopy(list)
        Nlist = len(list)
        chunkSize = float(Nlist/N)
        chunks = [ [] for _ in range(N)] 
        chunkSizeInt = int(chunkSize)

        if chunkSize < 1:
            remainder = Nlist
            chunkSizeInt = 0
            if self.root: print(' **Note: All PEs not being utilized** ')
        else:
            remainder = int((chunkSize % float(chunkSizeInt)) * N)

        for NN in np.arange(N):
            thisLen = chunkSizeInt
            if remainder > 0:
                thisLen += 1
                remainder -= 1
            for nn in np.arange(thisLen):
                chunks[NN].extend([newList.pop(0)])
        return chunks

    def init(self):
        #Serial Version
        t = time.time()
        self.root = True
        self.simList = []
        self.gridList = self.batch[0]

        bar = pb.ProgressBar(len(self.gridList))
        nn = 0
        for grd in self.gridList:
            #print('b = ' + str(self.gridLabels[nn]))
            self.simList.append(simulate(grd, self.env, self.N, findT = self.findT))
            bar.increment()
            bar.display()
            nn += 1
        bar.display(force = True)
        self.findLineStats()
        print('Elapsed Time: ' + str(time.time() - t))
        sys.stdout.flush()
        self.plotStats()

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

#Inputs: (self, batch, envs, N = 1000, findT = None, printOut = False)
#Public Methods: 
#Attributes: lines, lineStats
# For doing the same line from many angles, to get statistics


## Level 3: Initializes many Multisims, varying a parameter
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

        self.simulate_now()


    def simulate_now(self):
        comm = MPI.COMM_WORLD
        if self.root:
            print('\nCoronaSim!')
            print('Written by Chris Gilbert')
            print('-------------------------\n')

        if self.firstRunEver:
            self.count = 0
            self.batch = self.fullBatch
            if self.root and self.print: 
                self.bar = pb.ProgressBar(len(self.labels))

            self.sims = []
            self.profiles = []
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

            self.firstRunEver = False
            
            thisSim = multisim(thisBatch, self.envs, self.N, printOut = self.printMulti)
            comm.barrier()
            if self.root:
                self.sims.append(thisSim)
                self.profiles.append(thisSim.profiles)

                if self.print:
                    if self.printMulti: print('\nBatch Progress')
                    self.bar.increment()
                    self.bar.display(True)

                self.save()

        if self.root: 
            self.__findBatchStats()
            self.complete = True
            print('\nBatch Complete: '+ str(self.batchName))
            self.envs = []
            self.save(printout = False)

    def findRank(self):
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.root = self.rank == 0


    def doStats(self):
        self.__findSampleStats()
        self.makeVrms()
        self.plotStatsV()

    def redoStats(self):
        self.__findBatchStats()
        self.doStats()

    def __findBatchStats(self):
        self.lineStats = []
        for profiles in self.profiles:
            self.lineStats.append(self.__findSimStats(profiles))

    def __findSimStats(self, profiles):
        simStats = []
        for profile in profiles:
            simStats.append(self.__findProfileStats(profile))
        return simStats

    def __findProfileStats(self, profile):
        maxMoment = 5
        moment = np.zeros(maxMoment)
        for mm in np.arange(maxMoment):
                moment[mm] = np.dot(profile, self.env.lamAx**mm)

        power = moment[0] 
        mu = moment[1] / moment[0]
        sigma = np.sqrt(moment[2]/moment[0] - (moment[1]/moment[0])**2)
        skew = (moment[3]/moment[0] - 3*mu*sigma**2 - mu**3) / sigma**3
        kurt = (moment[4] / moment[0] - 4*mu*moment[3]/moment[0]+6*mu**2*sigma**2 + 3*mu**4) / sigma**4 - 3
        return [power, mu, sigma, skew, kurt]


    def __findSampleStats(self):
        #Does statistics on the statistics
        self.stat =  [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
        self.statV = [[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]]
        for vars, impact in zip(self.lineStats, self.labels):
                allAmp = [x[0] for x in vars]
                allMean = [x[1] for x in vars]
                allMeanC = [x[1] - self.env.lam0 for x in vars]
                allStd = [x[2] for x in vars]
                allSkew = [x[3] for x in vars]
                allKurt = [x[4] for x in vars]
            
                #Wavelength Units
                self.stat[0][0].append(np.mean(allAmp))
                self.stat[0][1].append(np.std(allAmp))

                self.stat[1][0].append(np.mean(allMeanC))
                self.stat[1][1].append(np.std(allMeanC))
            
                self.stat[2][0].append(np.mean(allStd))
                self.stat[2][1].append(np.std(allStd))

                self.stat[3][0].append(np.mean(allSkew))
                self.stat[3][1].append(np.std(allSkew))

                self.stat[4][0].append(np.mean(allKurt))
                self.stat[4][1].append(np.std(allKurt))

                #Velocity Units
                self.statV[0][0].append(np.mean(allAmp))
                self.statV[0][1].append(np.std(allAmp))

                self.statV[1][0].append(self.__mean2V(np.mean(allMean)))
                self.statV[1][1].append(np.std([self.__mean2V(x) for x in allMean]))
                
                T = self.env.interp_rx_dat(impact, self.env.T_raw)
                self.statV[2][0].append(self.__std2V(np.mean(allStd), T))
                self.statV[2][1].append(self.__std2V(np.std(allStd), T))
#TODO make these in velocity units
                self.statV[3][0].append(np.mean(allSkew))
                self.statV[3][1].append(np.std(allSkew))

                self.statV[4][0].append(np.mean(allKurt))
                self.statV[4][1].append(np.std(allKurt))

    def __mean2V(self, mean):
        return self.env.cm2km((self.env.ang2cm(mean) - self.env.ang2cm(self.env.lam0)) * self.env.c / 
                (self.env.ang2cm(self.env.lam0)))

    def __std2V(self, std, T):
        return self.env.cm2km(np.sqrt((np.sqrt(2) * self.env.ang2cm(std) * self.env.c / (self.env.ang2cm(self.env.lam0)))**2 - \
            (2 * self.env.KB * T / self.env.mi)))
                           
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

    def plotStatsV(self):
        f, axArray = plt.subplots(5, 1, sharex=True)
        f.canvas.set_window_title('Coronasim')
        doRms = True
        try:
            labels = np.asarray(self.doneLabels)
        except: 
            labels = np.arange(len(self.profiles))
            doRms = False
        mm = 0
        titles = ['Intensity', 'Mean Redshift', 'Line Width', 'Skew', 'Excess Kurtosis']
        ylabels = ['', 'km/s', 'km/s', '', '']
        thisBlist = self.Blist
        f.suptitle('Off Limb Line Statistics \n Wavelength: ' + str(self.env.lam0) + ' Angstroms\nLines per Impact: ' + str(self.Npt) + '\n Envs: ' + str(self.Nenv) + '; Lines per Env: ' + str(self.Nrot))
        for ax in axArray:
            if mm == 0: ax.set_yscale('log')
            ax.errorbar(labels, self.statV[mm][0], yerr = self.statV[mm][1], fmt = 'o')
            if mm == 2 and doRms:
                for vRms in self.vRmsList:
                    ax.plot(labels, vRms, label = str(thisBlist.pop(0)) + 'G')
                ax.legend(loc = 4)
            if mm == 1 or mm == 3 or mm == 4:
                ax.plot(labels, np.zeros_like(labels))
            ax.set_title(titles[mm])
            ax.set_ylabel(ylabels[mm])
            mm += 1
            ax.autoscale(tight = False)
        ax.set_xlabel(self.xlabel)
        grid.maximizePlot()
        plt.show()

    def save(self, batchName = None, keep = False, printout = True):

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


        script_dir = os.path.dirname(os.path.abspath(__file__))   
        absPath = os.path.join(script_dir, batchPath)  
        with open(absPath, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

        self.sims = sims
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



# For doing a multisim at many impact parameters
class impactsim(batchjob):
    def __init__(self, batchName, envs, Nb = 10, iter = 1, b0 = 1.05, b1= 1.50, N = (1500, 10000)):
        self.Nb = Nb
        self.batchName = batchName

        #Figure out appropriate number of rotlines per env
        comm = MPI.COMM_WORLD
        self.size = comm.Get_size()
        self.root = comm.Get_rank() == 0
        try:
            self.Nenv = len(envs)
        except: self.Nenv = 1

        #Lines per environment
        if self.root and self.size < self.Nenv:
            print('**Warning: More envs than PEs, will take additional time**')
        self.Nrot = np.floor(iter * max(1, self.size / self.Nenv))

        #Lines per impact
        self.Npt = self.Nrot * self.Nenv
        #Total Lines
        self.Ntot = self.Npt * self.Nb

        self.print = True
        self.printMulti = True

        self.N = N
        self.labels = np.round(np.linspace(b0,b1,Nb), 4)
        self.impacts = self.labels
        self.xlabel = 'Impact Parameter'    
        self.fullBatch = []
        for ind in self.impacts:
            self.fullBatch.append(grid.rotLines(N = self.Nrot, b = ind)) 

        super().__init__(envs)
        
        return
        
    def makeVrms(self):
        self.vRmsList = []
        self.Blist = np.linspace(0,30,5).tolist()

        for B in self.Blist:
            thisV = []
            for impact in self.impacts:
                thisV.append(self.env.cm2km(self.env.findVrms(impact, B)))
            self.vRmsList.append(thisV)

        #for vRms in self.vRmsList:
        #    plt.plot(self.labels, vRms)

        #plt.show()
        return
        
 
def loadBatch(batchName):
    slash = os.path.sep
    batchPath = '..' + slash + 'dat' + slash + 'batches' + slash + batchName + '.batch'
 
    absPth = absPath(batchPath)
    fail = False
    try:
        with open(absPth, 'rb') as input:
            return pickle.load(input)
    except:
        fail = True
    if fail:
        sys.exit('Batch Not found')

def restartBatch(batchName):
    myBatch = loadBatch(batchName)
    myBatch.findRank()
    myBatch.simulate_now()
    return myBatch

def plotBatch(batchName):
    myBatch = loadBatch(batchName)
    myBatch.findRank()
    myBatch.redoStats()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

def nothing():
    a = 1
    return a

