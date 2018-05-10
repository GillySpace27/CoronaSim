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
from itertools import cycle

#import chianti.core as ch
from scipy import signal as scisignal
from scipy import io
from scipy import ndimage
from scipy import interpolate as interp
from scipy.stats import norm
import scipy.stats as stats
from scipy.optimize import curve_fit, approx_fprime, minimize_scalar
import copy

#import warnings
#with warnings.catch_warnings():
#    warnings.simplefilter("ignore")
#    import astropy.convolution as con
#from numba import jit

from collections import defaultdict
from collections import OrderedDict
#import chianti.constants as const
#import chianti.util as cutil


import gridgen as grid
import progressBar as pb
import math
import time
import pickle
import glob
import warnings
warnings.simplefilter("ignore")

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
    def_2DLOSFile = os.path.join(datFolder, 'vxnew_2D_gilly40401.sav')
    def_ionFile = os.path.join(datFolder, 'useIons.csv')
    def_solarSpecFileHigh = os.path.abspath(os.path.join(datFolder, 'solarSpectrum/xIsun_whole.dat'))
    def_solarSpecFileLow = os.path.abspath(os.path.join(datFolder, 'solarSpectrum/EVE_L3_merged_2017347_006.fit'))
    #def_solarSpecFileLong = os.path.abspath(os.path.join(datFolder, 'solarSpectrum/recent.txt'))


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
    hergs = 6.626e-27 #ergs*sec
    hjs = 6.626e-34 #Joules*sec
    KB = 1.380e-16 #ergs/Kelvin
    K_ev = 8.6173303e-5 #ev/Kelvin
    r_Mm = 695.5 #Rsun in Mega meters
    mH = 1.67372e-24 #grams per hydrogen
    mE = 9.10938e-28 #grams per electron
    mP = 1.6726218e-24 #grams per proton
    amu = 1.6605e-24 #Grams per amu

    #Parameters
    rstar = 1
    B_thresh = 6.0
    fmax = 8.2
    theta0 = 28
    S0 = 7.0e5

    #for plotting
    lastPos = 1e8
    plotMore = True

    ##Element Being Observed
    #mI = 9.2732796e-23 #grams per Iron
    #ionString = 'fe'
    #element = 26
    #ion = 11
    #lower = 1 #lower energy level
    #upper = 38 #upper energy level

    ##LamAxis Stuff   #######################
    #Ln = 600
    #lam0 = 188.217
    #lamPm = 0.35

    psfSig = 0.047 #Angstroms


    def __init__(self, Bfile = None, bkFile = None, analyze = False, name = "Default", fFile = None):
        #Initializes
        self.name = name
        self._bfileLoad(Bfile, plot=False)
        self.__processBMap(thresh = 0.9, sigSmooth = 4, plot = False, addThresh = False)
        if analyze: self.analyze_BMap2()
        
        self._plasmaLoad(bkFile)
        self.spectrumLoad()
        self.expansionCalc()
        self.ionLoad()
        self._fLoad(fFile)
        self._hahnLoad()
        self._LOS2DLoad()

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
        print('')

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
        self.rx_raw = x[:,0] #Solar Radii
        self.rho_raw = x[:,1] #g/cm^3
        self.ur_raw = x[:,2] #cm/s
        self.vAlf_raw = x[:,3] #cm/s
        self.T_raw = x[:,4] #K


        if False:
            fig, (ax1) = plt.subplots(1, 1, True)

            ax2 = ax1.twinx()
            #ax2.set_ylabel('Density', color='r')
            ax2.tick_params('y', colors='r')

            lns1 = ax1.loglog(self.rx_raw, self.cm2km(self.ur_raw), label = 'Wind Speed (km/s)')
            lns2 = ax2.loglog(self.rx_raw, self.rho_raw, 'r:', label = 'Density (g/$cm^3$)')
            lns3 = ax1.loglog(self.rx_raw, self.cm2km(self.vAlf_raw), label = 'Alfven Speed (km/s)')
            lns4 = ax1.loglog(self.rx_raw, self.T_raw, label = 'Temperature (K)')

            #ax1.set_title('ZEPHYR Outputs')
            ax1.set_xlabel('r / $R_\odot$')


            lns = lns1+lns2+lns3+lns4
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc=0)



            #ax1.legend()
            #ax2.legend()
            ax1.set_xlim([1.001, 20])

            plt.show()

    def spectrumLoad(self):
        """Load the spectral data for the resonant profiles"""
        ##TODO find the correct absolute scaling for these measurements

        ###Load in the EVE data
        #from astropy.io import fits
        #hdulist = fits.open(self.def_solarSpecFileLow)
        #lamAx = hdulist[4].data['WAVELENGTH'].T.squeeze() #nm
        #units = hdulist[4].data['IRRADIANCE_UNITS'] #Watts/m^2/nm


        ##Average many of the EVE Spectra
        #intensity = np.zeros_like(hdulist[5].data[0]['SP_IRRADIANCE'])
        #averageDays = 365*2
        #succeed = 0
        #fail = 0
        #for ii in np.arange(averageDays):
        #    ints = hdulist[5].data[ii]['SP_IRRADIANCE']
        #    if -1 in ints: fail += 1; continue
        #    succeed += 1
        #    intensity += ints
        #    #if np.mod(succeed, 10) == 0: plt.plot(lamAx, ints, label = ii)
        #intensity /= succeed #Watts/m^2/nm

        #if False:
        #    plt.plot(lamAx, intensity,'k', label = 'Sum')
        #    plt.title("Tried: {}, Succeeded: {}, Failed: {}".format(averageDays, succeed, fail))
        #    plt.yscale('log')
        #    plt.show()
        #    #import pdb; pdb.set_trace()

        ##Attempt to put in correct units
        #lamAxLow = lamAx * 10 #angstrom
        #intensity #Watts/m^2/nm
        #solarSpecLow = intensity / 10 / 10000 /self.findSunSolidAngle(215) 
        #                #W/cm^2/sr/Angstrom

        ##Load in the SUMER Spectral Atlas
        x = np.loadtxt(self.def_solarSpecFileHigh, skiprows=13)
        lamAxHigh = x[:,0]  #Angstroms
        solarSpecHighRaw = x[:,1] #erg/s/cm2/sr/Angstrom 
        solarSpecHigh = solarSpecHighRaw #ergs/s/cm^2/sr/Angstrom 

        ###Concatenate the two measurements
        ##Find the kink
        #lowMax = np.max(lamAxLow)
        #jj = 0
        #while lamAxHigh[jj] < lowMax:
        #    jj += 1
        #newLamAxHigh = lamAxHigh[jj:]
        #newSolarSpecHigh = solarSpecHigh[jj:]

        ##Make them match at the kink
        #last = solarSpecLow[-1]
        #first = solarSpecHigh[jj]
        #ratio = first/last
        #newsolarSpecLow = solarSpecLow * ratio

        #Concatenate
        #self.solarLamAx = np.concatenate((lamAxLow, newLamAxHigh))
        #self.solarSpec = np.concatenate((newsolarSpecLow, newSolarSpecHigh))

        if False:
            plt.plot(self.solarLamAx, self.solarSpec)
            plt.plot(lamAxHigh, solarSpecHigh)
            plt.axvline(lowMax, c='k')
            plt.yscale('log')
            plt.xscale('log')
            plt.show()

        #Create primary deliverable: Interpolation object
        self.solarLamAx = lamAxHigh #Angstroms
        self.solarSpec = solarSpecHigh #ergs/s/cm^2/sr/Angstrom 
        self.solarInterp = interp.interp1d(lamAxHigh, solarSpecHigh)#, kind='cubic') #ergs/s/cm^2/sr/Angstrom
        pass
        


    def returnSolarSpecLam(self, lowLam, highLam):
        try:
            ll = 0
            while self.solarLamAx[ll] < lowLam:
                ll += 1
            lowInd = ll
            while self.solarLamAx[ll] < highLam:
                ll += 1
            highInd = ll
            return self.solarLamAx[lowInd:highInd], self.solarSpec[lowInd:highInd]
        except: raise IndexError 
        
    def returnSolarSpecLamFast(self, lowLam, highLam, lamAx, I0array):
        try:
            ll = 0
            while lamAx[ll] < lowLam:
                ll += 1
            lowInd = ll
            while lamAx[ll] < highLam:
                ll += 1
            highInd = ll
            return lamAx[lowInd:highInd], I0array[lowInd:highInd]
        except: raise IndexError 




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
        pass

    def fPlot(self, scale = 'linear'):
        fig, ax = plt.subplots()
        ax.plot(self.fr, self.f1_raw, label = 'f1 - Wind')
        ax.plot(self.fr, self.f2_raw, label = 'f2 - Waves')
        ax.plot(self.fr, self.f3_raw, label = 'f3 - Thermal')
        ax.set_title('Geometric Weighting Functions')
        ax.set_xlabel('Impact Parameter')
        ax.axhline(1, color ='k')
        ax.set_xscale(scale)
        ax.set_xlim([1.001, 5])
        ax.set_ylim([0.5,1.25])
        ax.legend()
        plt.show()

    def _xiLoad(self):
        ##Load Xi
        #self.xiFile1 = self.def_xiFile1 
        #self.xiFile2 = self.def_xiFile2 
        ##plt.figure()

        #x = np.loadtxt(self.xiFile1, skiprows=1)
        #self.xi1_t = x[:,0]
        #self.xi1_raw = x[:,1]
        #self.last_xi1_t = x[-1,0]
        ##plt.plot(self.xi1_t, self.xi1_raw)

        #y = np.loadtxt(self.xiFile2, skiprows=1)
        #self.xi2_t = y[:,0]
        #self.xi2_raw = y[:,1]
        #self.last_xi2_t = y[-1,0]

        #self.tmax = int(min(self.last_xi1_t, self.last_xi2_t))
        ##plt.plot(self.xi2_t, self.xi2_raw)
        ##plt.show()
        self.xi1_t = self.R_time
        self.last_xi1_t = self.xi1_t[-1]
        self.tmax = self.last_xi1_t

        xi = self.R_vlos[-1,:].flatten()
        xistd = np.std(xi)
        xinorm = xi / xistd
        self.xi1_raw = xinorm
        pass

    def _hahnLoad(self):
        x = np.loadtxt(self.def_hahnFile)
        self.hahnAbs = x[:,0]
        line1 = x[:,1]
        line2 = x[:,2]

        self.hahnPoints = line1
        self.hahnError = line2-line1

    def _LOS2DLoad(self):
        x = io.readsav(self.def_2DLOSFile)
        self.R_ntime = x.ntime
        self.R_nzx = x.nzx
        self.R_time = x.time
        self.R_zx = x.zx
        self.R_vlos = x.vxlos
        self._xiLoad()

  ## Ion Stuff ##########################################################################

    def _chiantiLoad(self):
        """Load the chianti info for each of the desired ions"""

        #Load Chianti Data for every ion
        for ion in self.ions:

            ##Load things from files###############

            #Load the ionization fraction file
            self.cLoadIonFraction(ion)

            #Load in elemental abundance info 
            self.cLoadAbund(ion)

            #Load in upsilon(T) info 
            self.cLoadUpsilon(ion)

            #Load in statistical weights 
            #self.cLoadStatWeight(ion) #Depricated

            #Load in Angular Momenta and Stat Weights
            self.cLoadAngularMomentum(ion)

            #Load the Einstein coefficients
            self.cLoadEinstein(ion)

            #Load the Ionization Potential
            self.cLoadIonization(ion)

            #Load the Recombination rate TO this ion
            self.cLoadRecombination(ion)

            #Load the collision rate
            self.cLoadCollision(ion)

            ##Do Calculations######################

            #Make the nGrid for this Element
            self.simulate_ionization(ion)

            #Find the freezing density
            self.findFreeze2(ion)
            
            ##Make the Spectral Irradiance arrays
            self.makeIrradiance(ion)

            #Check if there are multiple lines here
            #self.checkOverlap(ion)
          
            if False: #Plot the info
                fig, ax1 = plt.subplots()
                #fig.subplots_adjust(right=0.8)
                ax2 = ax1.twinx()

                ax2.plot(ion['chTemps'], ion['chFracs'], 'bo-')
                ax2.set_xlabel('Temperature')
                ax2.set_ylabel('Fraction', color='b')
                ax2.tick_params('y', colors='b')

            
                ax1.plot(ion['splinedUpsX'], ion['splinedUps'], 'ro-')
                ax1.set_xlabel('Temperature')
                ax1.set_ylabel('Upsilon', color='r')
                ax1.tick_params('y', colors='r')

                plt.title("Data for {}_{}:{}->{} \n $\lambda$ = {}".format(ion['ionString'], ion['ion'], ion['upper'], ion['lower'],ion['lam00']))

                height = 0.9
                left = 0.09
                plt.figtext(left, height + 0.06, "Abundance: {:0.4E}".format(ion['abundance']))
                plt.figtext(left, height + 0.03, "Weight: {}".format(ion['wi']))
                plt.show()
            pass     

    def cLoadIonFraction(self, ion):
        """Load in the ionization fracion in equilibrium as per Chianti"""

        #Load in the ionization fraction info
        chi = np.loadtxt(self.def_ioneq)
        for idx in np.arange(len(chi[:,0])):
            if chi[idx,0] == ion['element'] and chi[idx,1] == ion['ion']: break
        else: raise ValueError('{}_{} Not Found in fraction file'.format(ion['ionString'], ion['ion']))

        ion['chTemps'] = chi[0,2:]
        ion['chFracs'] = chi[idx,2:] + 1e-100
        return ion['chTemps'], ion['chFracs']

    def cLoadAbund(self, ion):
        """Load the elemental abundance for this element"""
        abund = np.loadtxt(self.def_abund, usecols=[1])
        ion['abundance'] = 10**(abund[ion['element']-1]-abund[0])
        return ion['abundance']

    def cLoadUpsilon(self, ion):
        """Load in the upsilon(T) file"""
        fullstring = ion['ionString'] + '_' + str(ion['ion'])
        ionpath = (self.def_ionpath +'/'+ ion['ionString'] + '/' + fullstring)
        fullpath = os.path.normpath(ionpath + '/'+ fullstring + '.scups')

        getTemps = False
        getUps = False
        try:
            with open(fullpath) as f:
                for line in f:
                    data = line.split()
                    if getUps == True:
                        ion['ups'] = [float(x) for x in data]
                        break

                    if getTemps == True:
                        ion['upsTemps'] = [float(x) for x in data]
                        getUps = True
                        continue

                    if data[0] == str(ion['lower']) and data[1] == str(ion['upper']):
                        ion['upsInfo'] = [float(x) for x in data]
                        getTemps = True
            
            ion['splinedUpsX'] = np.linspace(0,1,200)
            ion['splinedUps'] = interp.spline(ion['upsTemps'], ion['ups'], ion['splinedUpsX'])
            #ion['gf'] = ion['upsInfo'][3]
        except: raise ValueError('Transition {}_{}:{}->{} not found in scups file'.format(ion['ionString'], ion['ion'], ion['upper'], ion['lower']))
        return ion['splinedUpsX'], ion['splinedUps'] 

    def cLoadAngularMomentum(self,ion):
        """Load in the angular momenta for the upper and lower levels"""
        fullstring = ion['ionString'] + '_' + str(ion['ion'])
        ionpath = (self.def_ionpath +'/'+ ion['ionString'] + '/' + fullstring)       
        fullpath2 = os.path.normpath(ionpath + '/'+ fullstring + '.elvlc')   
        found1 = False
        found2 = False
        with open(fullpath2) as f2:     
            for line in f2:
                data = line.split()
                if data[0] == str(ion['lower']):
                    ion['JL'] = float(data[-3])
                    found1 = True
                if data[0] == str(ion['upper']):
                    ion['JU'] = float(data[-3])
                    found2 = True
                if found1 and found2: break
        if not found1: raise ValueError('Angular Momemntum {}_{}:{} not found in elvlc file'.format(ion['ionString'], ion['ion'], ion['lower']))
        if not found2: raise ValueError('Angular Momemntum {}_{}:{} not found in elvlc file'.format(ion['ionString'], ion['ion'], ion['upper']))

        ion['dj'] = ion['JU'] - ion['JL']
        j = ion['JL']

        if ion['dj'] == 1: ion['E1'] = ( (2*j+5) * (j + 2) ) / ( 10*(j+1)*(2*j+1) )
        elif ion['dj'] == 0: ion['E1'] = ( (2*j-1) * (2*j+3) ) / ( 10*j * (j+1) )
        elif ion['dj'] ==-1: ion['E1'] = ( (2*j-3) * (j - 1) ) / ( 10*j * (2*j+1) )
        else: raise ValueError('Bad Change in Momentum')

        #Depricated, old style E
        ion['E'] = 1- ion['E1']/4

        #Convert the momenta into statistical weights
        ion['wiU'] = 2 * ion['JU'] + 1
        ion['wi'] = 2 * ion['JL'] + 1 


        #from fractions import Fraction
        #print('{}, {} E1 = {} = {}; E = {}'.format(fullstring, ion['lam00'], ion['E1'], str(Fraction(ion['E1']).limit_denominator()), str(Fraction(ion['E']).limit_denominator())))
        
        return ion['E1']

    def cLoadEinstein(self, ion):
        """Load in the Einstein Coefficients and the Chianti line wavelengths"""
        fullstring = ion['ionString'] + '_' + str(ion['ion'])
        ionpath = (self.def_ionpath +'/'+ ion['ionString'] + '/' + fullstring)
        fullpath3 = os.path.normpath(ionpath + '/'+ fullstring + '.wgfa')
        found = False
        with open(fullpath3) as f3:     
            for line in f3:
                data = line.split()
                if data[0] == str(ion['lower']) and data[1] == str(ion['upper']):
                    ion['lam00'] = float(data[2]) #angstroms
                    ion['nu00'] = self.c / self.ang2cm(ion['lam00']) #1/s
                    ion['gf'] = float(data[3]) #dimensionless
                    ion['A21'] = float(data[4]) # inverse seconds
                    found = True
                    break
        if found == False: raise ValueError('Einstein Coefficient for {}_{}:{} not found in wgfa file'.format(ion['ionString'], ion['ion'], ion['lower']))
           
        ion['B21'] = ion['A21'] * self.c**2 / (2*self.hergs*ion['nu00']**3) #1/s * cm^2 / s^2 * 1/(ergs *s) * s^3 = cm^2 / (erg * s)
        ion['B12'] = ion['B21'] * ion['wiU'] / ion['wi'] #cm^2 / (erg*s)

        #print("A = {}, B21 = {}".format(ion['A21'], ion['B21']))
        #print('WiL: {}, WiU: {}'.format(ion['wi'], ion['wiU']))

    def checkOverlap(self, ion):
        '''See if there are any other lines of this ion nearby'''
        fullstring = ion['ionString'] + '_' + str(ion['ion'])
        ionpath = (self.def_ionpath +'/'+ ion['ionString'] + '/' + fullstring)
        fullpath3 = os.path.normpath(ionpath + '/'+ fullstring + '.wgfa')

        #Check for multiple overlapping lines
        dL = ion['lamPm']
        waveMin = ion['lam00'] - dL
        waveMax = ion['lam00'] + dL
        matchList = []
        with open(fullpath3) as f3:     
            for line in f3:
                data = line.split()
                if len(data) < 2: break
                wavelength = float(data[2])
                if waveMin < wavelength < waveMax:
                    if int(data[0]) > 2: continue
                    delta = (float(data[2])-ion['lam00']) / ion['I0Width']
                    matchList.append([int(data[0]), int(data[1]), float(data[2]), float('{:0.5}'.format(delta)) ]) #angstroms
        print('{}, num= {}\n {}'.format(fullstring, len(matchList), matchList))
        #print(ion['I0Width'])


    def cLoadIonization(self, ion):
        """Find the ionization potential for this ion""" #TODO Check this is right
        ionizationPath = os.path.normpath('{}/ip/chianti.ip'.format(self.def_ionpath))

        with open(ionizationPath) as f:
            for line in f:
                data_raw = line.split()
                data = [float(x) if '.' in x else int(x) for x in data_raw]
                    
                if data[0] == ion['element']:
                    if data[1] == ion['ion']:
                        thisE = data[2] #cm^-1
                    if data[1] == ion['ion'] + 1:
                        nextE = data[2] #cm^-1
                if data[0] > ion['element']:
                    break

        hc = 1.9864458e-16 #erg * cm
        ion['ionizationPotential'] = np.abs(thisE - nextE) * hc #ergs


    def cLoadOneRecombRate(self, ionString, ionNum):
        '''Load in the recombination rate from ion ionNum, as a function of temperature'''
        higherString = "{}_{}".format(ionString, ionNum)
        ionpath = "{}/{}/{}/{}.rrparams".format(self.def_ionpath, ionString, higherString, higherString)
        fullpath = os.path.normpath(ionpath)
        # print('this is {} {}'.format(ionString,ionNum))
        try:
            with open(fullpath) as f:
                ii = 0
                lines = []
                for line in f:
                    if ii < 2: lines.append(line)
                    ii += 1
            type = int(lines[0].split()[0])
            params_raw = lines[1].split()[2:]
            params = [float(x) if '.' in x else int(x) for x in params_raw]

            if type == 1:
                params.pop(0)
                recomb_func = partial(self.recomb1, *params)
            elif type == 2:
                params.pop(0)
                recomb_func = partial(self.recomb2, *params)
            elif type == 3:
                recomb_func = partial(self.recomb3, *params)
            # else: notreal += 1
                
            recomb_T = np.logspace(3,8,100) # Kelvin
            recomb = np.asarray([recomb_func(T) for T in recomb_T]) #cm^3/s

            #newx = np.logspace(3,8,500)
            #plt.plot(newx, [self.interp_recomb(ion, T) for T in newx])
            #plt.plot(ion['recomb_T'], ion['recomb'], 'ko')
            #plt.yscale('log')
            #plt.show()

        except: raise ValueError('Recombination {}: {}->{} not found in rrparams file'.format(ionString, ionNum, ionNum-1))
        return recomb_T, recomb
        
    def cLoadRecombination(self, ion):
        '''Load in the recombination rate to this ion'''
        ion['recomb_T'], ion['recomb'] = self.cLoadOneRecombRate(ion['ionString'], ion['ion']+1)
        return ion['recomb_T'], ion['recomb']

    def cLoadOneCollisRate(self, thisElement, ionNum):
        """Load in the collisional ionization rate as F(T) out of ion ionNum"""
        path = os.path.join(self.datFolder, 'collisions.dat')
        with open(path) as f:
            for line in f:
                newLine = ''
                for letter in line:
                    if letter is 'D':
                        letter = 'E'
                    newLine = newLine + letter

                data = newLine.split('/')
                info = data[0].split(',')[1:3]
                info[1] = info[1][:-1]
                info = [int(x) for x in info]
                info[1] = info[0] - info[1] + 1

                element = info[0]
                ionum = info[1]
                values = [float(x) for x in data[1].split(',')]

                if element == thisElement and ionum == ionNum:
                    break

        #print('{}_{}: {}'.format(element, ionum, values))

        collis_T = np.logspace(3,8, 100) #Kelvin
        cFit = partial(self.collisFit, *values)
        collis = np.asarray([cFit(T) for T in collis_T])   #cm^3 /s  
        return collis_T, collis

    def cLoadCollision(self, ion):
        """Load the collision rate as F(T)"""
        ion['collis_T'], ion['collis'] = self.cLoadOneCollisRate(ion['element'], ion['ion'])
        return ion['collis'], ion['collis_T']

    def makeIrradiance(self, ion):
        """Crop the appropriate section of the irradiance array"""
        rez = 1000 #int(2*ion['ln'])
        lam0 = ion['lam00']

        #First approximation, cut out the spectral line
        pm = 0.003 * lam0 #2 * ion['lamPm']
        if lam0 < 700:
            pm *= 2
        if lam0 < 225:
            pm *= 2
        lamAxPrime, I0array = self.returnSolarSpecLam(lam0-pm, lam0+pm) #ergs/s/cm^2/sr/Angstrom
        #lamAxPrime = np.linspace(lam0-pm, lam0+pm, rez)
        #I0array = self.solarInterp(lamAxPrime) 

        #Fit a Gaussian to it
        fits, truncProfile = self.simpleGaussFit(I0array, lamAxPrime, lam0)

        #Find the wind limits
        vFast = self.interp_wind(40)

        lamPm = vFast / self.c * lam0
        lamHigh = lam0+lamPm*2
        lamLow = lam0-lamPm*2

        ion['I0Width'] = fits[2]

        spread = 8
        highLimit = lamHigh + spread*ion['I0Width']
        #highLimit = lam0 + spread*fits[2]
        #lowLimit = lam0 - spread*fits[2]
        lowLimit = lamLow - spread*ion['I0Width']

        #Store the Irradiance array
        #Watts/cm^2/sr/Angstrom
        ion['lamAxPrime'], ion['I0array'] = self.returnSolarSpecLam(lowLimit, highLimit)
        ion['I0interp'] = interp.interp1d(ion['lamAxPrime'], ion['I0array'])#, kind='cubic') #ergs/s/cm^2/sr/Angstrom
        #ion['lamAxPrime'] = np.linspace(lowLimit, highLimit, rez)
        #ion['I0array'] = self.solarInterp(ion['lamAxPrime']) 



        ion['nu0'] = self.c /self.ang2cm(lam0)
        ion['nuAx'] = self.lamAx2nuAx(ion['lamAx'])
        ion['nuAxPrime'] = self.lamAx2nuAx(ion['lamAxPrime'])
        ion['nuI0array'] = self.lam2nuI0(ion['I0array'], ion['lamAxPrime'])



        #plt.plot(ion['nuAxPrime'], ion['nuI0array'])
        #plt.title("{}_{}".format(ion['ionString'], ion['ion']))
        #plt.show()
        if False:
            #Plot the irradiance array stuff
            plt.plot(lamAxPrime, I0array)
            plt.plot(lamAxPrime, self.gauss_function(lamAxPrime,*fits))
            #plt.plot(lamAxPrime, truncProfile)
            plt.axvline(lam0, c='grey')
            plt.axvline(lamHigh, c='grey', ls= ':')

            plt.axvline(highLimit, c='k', ls= '-')
            plt.axvline(lowLimit, c='k', ls= '-')

            plt.plot(ion['lamAxPrime'], self.gauss_function(ion['lamAxPrime'],fits[0], lamHigh,fits[2], fits[3]))
            
            plt.axvline(lamLow, c='grey', ls= ':')
            plt.axvline(lamLow - spread*fits[2], c='grey', ls= '--')
            plt.plot(ion['lamAxPrime'], self.gauss_function(ion['lamAxPrime'],fits[0], lamLow,fits[2], fits[3]))
            
            plt.plot(ion['lamAxPrime'], ion['I0array'], "c.", lw=3)
            plt.title("{}_{}".format(ion['ionString'], ion['ion']))
            plt.show()

    def lamAx2nuAx(self, lamAx):
        """Change a lamAx (ang) to a nuAx"""
        return self.c / self.ang2cm(lamAx)

    def lam2nuI0(self, I0, lamAx):
        """Change a ligt profile to frequency units"""
        nuAx = self.lamAx2nuAx(lamAx)
        return I0 * lamAx / nuAx

    def simpleGaussFit(self, profile, lamAx, lam0):
        sig0 = 0.2
        amp0 = np.max(profile) - np.min(profile)

        jj = 0
        while lamAx[jj] < lam0: jj+=1
        
        low = np.flipud(profile[:jj])
        high = profile[jj:]

        lowP = []
        last = low[0]
        for p in low:
            if p < last: lowP.append(p); last = p
            else: lowP.append(last)
        lowP = np.flipud(lowP)

        highP = []
        last = high[0]
        for p in high:
            if p < last: highP.append(p); last = p
            else: highP.append(last)

        

        truncProfile = np.concatenate((lowP, highP))
       
        popt, pcov = curve_fit(self.gauss_function, lamAx, truncProfile, p0 = [amp0, lam0, sig0, 0])
        amp = popt[0]
        mu = popt[1] #- lam0
        std = popt[2] 
        b = popt[3]
        area = popt[0] * popt[2]  
        return (amp, mu, std, b), truncProfile

    def gauss_function(self, x, a, x0, sigma, b):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))+b

    def collisFit(self, dE, P, A, X, K, T):
        U = dE / (self.K_ev * T)
        return A*(1+P*np.sqrt(U))/(X+U)*U**K*np.exp(-U) #cm^3/s

    def recomb1(self, A,B,T0,T1, T): 
        TT0 = np.sqrt(T/T0)
        TT1 = np.sqrt(T/T1)
        return A / ( TT0 * (1+TT0)**(1-B) * (1+TT1)**(1+B) )

    def recomb2(self, A,B,T0,T1,C,T2,T): 
        b = B + C*np.exp(-T2/T)
        TT0 = np.sqrt(T/T0)
        TT1 = np.sqrt(T/T1)
        return A / ( TT0 * (1+TT0)**(1-b) * (1+TT1)**(1+b) )

    def recomb3(self, A, eta, T):
        return A * (T/ 10000)**(-eta)

  
    def findSunSolidAngle(self, rx):
        """Return the proportion of the sky covered by the sun"""
        return 0.5*(1-np.sqrt(1-(1/rx)**2))


    def particleTimes(self,ion,R):
        """Determine the Collision and the Expansion time at a Radius"""
        rs_cm = self.r_Mm * 10**8

        rho = self.interp_rho(R)

        f = self.interp_rho
        e = 0.02
        drdrho = (f(R+e) - f(R))/(e*rs_cm)
        T = self.interp_T(R)
        nE = rho*0.9/self.mP

        recomb = self.interp_recomb(ion, T)
        collis = self.interp_collis(ion, T)
        wind = self.interp_wind(R)
        t_C = 1/(nE*(recomb+collis)) #Collisional Time
        t_E = np.abs(rho/(wind*drdrho)) #Expansion Time

        return t_C, t_E

    def particleTimeDiff(self,ion,R):
        """Determine the particle time difference"""
        t_C, t_E = self.particleTimes(ion, R)
        return np.abs(t_C - t_E)

    def findFreeze2(self,ion):
        """Determine the freezing radius and rho and T for an ion"""
        #Actually determine r_freeze
        func = partial(self.particleTimeDiff, ion)
        rez = minimize_scalar(func,method = 'Bounded', bounds = [1.0002,10])
        r_freeze = rez.x

        ion['r_freeze'] = r_freeze
        ion['rhoCrit'] = self.interp_rho(r_freeze)
        ion['TCrit'] = self.interp_T(r_freeze)

        #Plot
        if False:
            fig, ax2 = plt.subplots()

            rr = np.linspace(1.05, 10, 1000)
            tt1 = []
            tt2 = []
            for r in rr:
                a,b = self.particleTimes(ion, r)
                tt1.append(a)
                tt2.append(b)

            ax2.axvline(r_freeze)
            ax2.plot(rr, tt1, label = 'Collision')
            ax2.plot(rr, tt2, label = 'Expansion')
            ax2.legend()
            ax2.set_title('Frozen in at R = {:.4}'.format(r_freeze))
            fig.suptitle('Finding the Freezing Height/Density for Ion: {}_{}'.format(ion['ionString'], ion['ion']))
            ax2.set_yscale('log')
            ax2.set_xlabel('Impact Parameter')
            ax2.set_ylabel('Time')
            plt.show()

   


    def ionLoad(self):
        """Read the spreadsheet indicating the ions to be simulated"""
        self.ions = []
        import csv
        with open(self.def_ionFile) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if '#' in row['ionString']:
                    continue
                for key in row.keys():
                    if not key.casefold() == 'ionstring':
                        if '.' in row[key]: row[key] = float(row[key])
                        else: row[key] = int(row[key])
                row['lamAx'] = self.makeLamAxis(row['ln'], row['lam0'], row['lamPm'])
                self.ions.append(row)
        self._chiantiLoad()

    def plot2DV(self):
        fig, ax = plt.subplots()
        mean = np.mean(self.R_vlos.flatten())
        std = np.std(self.R_vlos.flatten())
        sp = 4
        vmin = mean - sp*std
        vmax = mean + sp*std
        im = ax.pcolormesh(self.R_time, self.R_zx + 1, self.R_vlos, vmin = vmin, vmax = vmax, cmap = 'RdBu')
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label('km/s')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('$R_{\odot}$')
        ax.set_title('Alfven Waves from Braid Code')
        plt.tight_layout()
        plt.show()

    def simulate_ionization(self, ion):

        #Load Recombination/Ionization Rates
        try: self.elements
        except: self.elements = dict()

        if not ion['ionString'] in self.elements:
            self.elements[ion['ionString']] = element(self, ion)
            #self.elements[ion['ionString']].plotAll()


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
        self.validMask = self.BMap_raw != 0
        self.label_im *= self.validMask
        self.voroBMap *= self.validMask
        self.BMap = self.voroBMap  

        rawTot = np.nansum(np.abs(self.BMap_raw))
        proTot = np.nansum(self.BMap)
        bdiff = np.abs(rawTot-proTot)
        #print("\nThe total raw field is {:0.4}, and the total processed field is {:.4}".format(rawTot, proTot))
        #print("The ratio of processed over raw is {:.4}".format(proTot/rawTot))


        if False:
            hist, edges = np.histogram(self.Bmap_means, 25)
            numLess = len([x for x in np.abs(self.Bmap_means) if x < 2])
            edges = edges[0:-1]
            fig, ax = plt.subplots()
            ax.step(edges, hist)
            ax.set_xlabel("Gauss")
            ax.set_ylabel('Number of Cells')
            ax.set_title('With Abs, Sum = {}, lessThan = {}'.format(np.sum(hist), numLess))
            plt.show()


        #fig, ax2 = plt.subplots()
        #pCons = self.mask(ax2, self.bFluxCons)
        #i5 = ax2.imshow(pCons, cmap = "RdBu", aspect = 'auto')
        #plt.colorbar(i5, ax = [ax2], label = "Percentage")
        #plt.show()

        if False: #Plot Slice of Map
            fig, ax0 = plt.subplots()
            f2, ax = plt.subplots()

            #Detect Color Range
            f = 5
            st1 = np.std(self.voroBMap.flatten())
            m1 = np.mean(self.voroBMap.flatten())
            vmin = 0#m1 - f*st1
            vmax = m1 + f*st1


            #Plot Raw Field Map
            image = label_im + 37
            newBmap = self.mask(ax0, image) 
            i0 = ax0.imshow(newBmap, cmap = 'prism', aspect = 'auto', vmin = 0)#, vmin = vmin, vmax = vmax)
            plt.colorbar(i0, ax=[ax0], label='Index')#, extend = 'max')
            ax0.set_title('Raw Magnetic Field')

            self.plotEdges(ax0, label_im)

            #Plot the Voromap
            newVoroBMap = self.mask(ax, self.voroBMap) 
            i2 = ax.imshow((newVoroBMap), cmap = 'magma', interpolation='none', aspect = 'equal')#, vmin = vmin, vmax = vmax)
            ax.set_title('Final Field Map')

            self.plotEdges(ax, self.label_im)

            #Plot the maxima points
            coordinates = []
            for co in coord: coordinates.append(co[::-1])
            for co in coordinates:
                ax0.plot(*co, marker = 'o', markerfacecolor='r', markeredgecolor='k', markersize = 6)
                ax.plot(*co, marker = 'o', markerfacecolor='w', markeredgecolor='k', markersize = 6)

            plt.tight_layout()
            plt.colorbar(i2, ax=[ax], label='Gauss')#, extend = 'max')
            plt.show()
            pass
        return

    def plotEdges(self, ax, map, domask = True):
            #Crude Edge Detection
            
            #map = self.label_im
            mask = np.zeros_like(map, dtype='float')
            width, height = mask.shape
            for y in np.arange(height):
                for x in np.arange(width):
                    if x == 0 or x >= width-1 or y == 0 or y >= height-1:
                        mask[x,y] = np.nan
                    else:
                        notedge = map[x, y] == map[x+1,y] 
                        if notedge: notedge = map[x, y] == map[x,y+1]

                        if notedge:
                            mask[x,y] = np.nan
                        else: mask[x,y] = 1

            #Plot the Edges
            my_cmap = copy.copy(plt.cm.get_cmap('gray')) # get a copy of the gray color map
            my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values
            if domask:
                newMask = self.mask(ax, mask) 
                ax.imshow(newMask, cmap = my_cmap, aspect = 'auto')
            else: ax.imshow(mask, cmap = my_cmap, aspect = 'auto')



    def mask(self, ax, array):
        almostmasked = np.asarray(array, 'float')
        almostmasked[~self.validMask] = np.nan
        masked = np.ma.masked_invalid(almostmasked)
        #ax.patch.set(hatch='x', edgecolor='black')
        return masked

    def __voronoify_sklearn(self, I, seeds, data):
        #label im, coords, bdata
        import sklearn.neighbors as nb
        #Uses the voronoi algorithm to assign stream labels
        tree_sklearn = nb.KDTree(seeds)
        pixels = ([(r,c) for r in range(I.shape[0]) for c in range(I.shape[1])])
        d, pos = tree_sklearn.query(pixels)
        cells = defaultdict(list)

        for i in range(len(pos)):
            cells[pos[i][0]].append(pixels[i])

        I2 = I.copy() #Index number
        I3 = I.copy().astype('float32') #Mean Flux
        I4 = I.copy().astype('float32') #Flux Difference
        label = 0
        self.Bmap_means = []
        for idx in cells.values():
            idx = np.array(idx)
            label += 1

            mean_col = data[idx[:,0], idx[:,1]].mean() #The mean value in the cell
            self.Bmap_means.append(mean_col)
            #npix = len(data[idx[:,0], idx[:,1]]) # num of pixels
            #sum_col = data[idx[:,0], idx[:,1]].sum() #The sum of the values in the cell
            #meanFlux = mean_col * npix    
            #rawFlux = sum_col  
            #bdiff = (meanFlux - rawFlux)/rawFlux #How much each cell is getting the flux wrong 

            I2[idx[:,0], idx[:,1]] = label 
            I3[idx[:,0], idx[:,1]] = mean_col
            #I4[idx[:,0], idx[:,1]] = bdiff

        return I2, label, I3 #, I4

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

    def plotBmap(self):
        fig = plt.figure()

        map = self.BMap_raw
        map[map == 0] = np.nan
        map = np.ma.masked_invalid(map)

        p0 = plt.pcolormesh(self.BMap_x, self.BMap_y, map, cmap = 'binary')
        ax = plt.gca()
        #ax.patch.set(hatch='x', edgecolor='black')
        ax.set_aspect('equal')
        plt.show()

    def plotXi(self):
        fig, (ax0, ax1) = plt.subplots(2,1,True)

        ax0.plot(self.xi1_t, self.xi1_raw)
        ax1.plot(self.xi2_t, self.xi2_raw)
        ax1.set_xlabel('Time (s)')
        #ax0.set_ylabel()
        fig.text(0.04, 0.5, 'Wave Velocity (km/s)', va='center', rotation='vertical')

        plt.show()


    def expansionCalc(self):
        """Calculate the super-radial expansion from the Zephyr Inputs"""
        
        #Get B from Zephyr
        self.B_calc = np.sqrt(4*np.pi*self.rho_raw)*self.vAlf_raw
        self.A_calc = 1/self.B_calc #Calculate flux tube area from that
        A0 = self.A_calc[0]
        F_raw = self.A_calc/(A0 * self.rx_raw**2) #Compare flux tube area to radial expansion

        #Normalize the top
        F_norm = F_raw / F_raw[-1] * self.fmax 

        #Truncate the bottom
        trunc = 0.855
        fcalc2 = copy.deepcopy(F_norm)
        fcalc2[fcalc2 < trunc] = trunc

        #Normalize the top and bottom
        scaleF = (fcalc2 - np.min(fcalc2)) / (np.max(fcalc2) - np.min(fcalc2))
        reScaleF = scaleF * (self.fmax -1) + 1

        #Save variable
        self.F_calc = reScaleF

        #Plotting
        if False:
            #plt.plot(self.rx_raw-1, F_norm, '--', label = "Calculated")
            #plt.plot(self.rx_raw-1, fcalc2,":", label = "Truncated")
            plt.plot(self.rx_raw-1, reScaleF, label = "Scaled")
            plt.plot(self.rx_raw-1, [self.getAreaF_b(b) for b in self.rx_raw], label = "Old Fit Function")


            plt.legend()
            plt.xscale('log')
            #plt.xlim([0,50])
            plt.xlabel("$r/R_\odot$ - 1")
            plt.ylabel("Superadial expansion factor $f(r)$")
            plt.show()

    def getAreaF_b(self, r):
        Hfit  = 2.2*(self.fmax/10.)**0.62
        return self.fmax + (1.-self.fmax)*np.exp(-((r-1.)/Hfit)**1.1)

    def getAreaF(self, r):
        return self.interp_rx_dat(r, self.F_calc)


  ## Light ############################################################################

    def makeLamAxis(self, Ln = 100, lam0 = 200, lamPm = 0.5):
        return np.linspace(lam0 - lamPm, lam0 + lamPm, Ln)

    def getIons(self, maxIons):
        if maxIons > len(self.ions):
            return self.ions
        return self.ions[0:maxIons]

    def findQt(self, ion, T):
        #Chianti Stuff
        Iinf = 2.18056334613e-11 #ergs, equal to 13.61eV
        kt = self.KB*T
        dE = self.ryd2erg(ion['upsInfo'][2])
        upsilon = self.findUpsilon(ion,T)
        return 2.172e-8*np.sqrt(Iinf/kt)*np.exp(-dE/kt)* upsilon / ion['wi'] 

    def findQtIonization(self, ion, T):
        #
        #Chianti Stuff
        Iinf = 2.18056334613e-11 #ergs, equal to 13.61eV
        kt = self.KB*T
        dE = ion['ionizationPotential']
        upsilon = self.findUpsilon(ion,T)
        return 2.172e-8*np.sqrt(Iinf/kt)*np.exp(-dE/kt)* upsilon / ion['wi']

    def findUpsilon(self, ion, T):
        Eij = ion['upsInfo'][2] #Rydberg Transition energy
        K = ion['upsInfo'][6] #Transition Type
        C = ion['upsInfo'][7] #Scale Constant

        E = np.abs(T/(1.57888e5*Eij))
        if K == 1 or K == 4: X = np.log10((E+C)/C)/np.log10(E+C)
        if K == 2 or K == 3: X = E/(E+C)

        Y = self.interp_upsilon(X, ion)
        if K == 1: Y = Y*np.log10(E+np.exp(1))
        if K == 3: Y = Y/(E+1)
        if K == 4: Y = Y*np.log10(E+C)

        return Y

    def findIonFrac(self, ion, T, rho):
        """Find the frozen-in ion density"""
        power = 6

        beta = rho/ion['rhoCrit']
        epsilon = beta**power/(1+beta**power)

        F_T = self.interp_frac(T, ion)
        F_F = self.interp_frac(ion['TCrit'], ion)

        if not self.ionFreeze: return F_T
        else: 
            return epsilon*F_T + (1-epsilon)*F_F
            #return F_T**epsilon * F_F**(1-epsilon)

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

    def __find_nearest(self,array,value):
        #Returns the index of the point most similar to a given value
        array = array-value
        np.abs(array, out = array)
        return array.argmin()

    def find_crossing(self, xx, A1, A2):
        A3 = np.abs(np.asarray(A1)-np.asarray(A2))
        return xx[A3.argmin()]

    #def __find_nearest(self,array,value):
    #    idx = np.searchsorted(array, value, side="left")
    #    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
    #        return idx-1
    #    else:
    #        return idx


    def findvRms(self, rr):
        #RMS Velocity
        #S0 = 7.0e5 #* np.sqrt(1/2)
        #return np.sqrt(S0*self.vAlf/((self.vPh)**2 * 
        #    self.rx**2*self.f*self.rho))
        if rr < 3:
            return self.extractVrms(rr)
        else:
            vTop = 6090764.71206 - 25000
            rTop = 2.98888888889
            return vTop*(rr/rTop)**-0.3

    def extractVrms(self, b):
        try:
            bInd = np.searchsorted(self.R_zx + 1, b) -1
            R_Array = self.R_vlos[bInd,:].flatten()
            V = np.std(R_Array)
            return V * 1e5
        except: return np.nan
        
    def interp_T(self, rx):
        return self.interp_rx_dat_log(rx, self.T_raw) #K

    def interp_rho(self, rx):
        return self.interp_rx_dat_log(rx, self.rho_raw) #g/cm^3

    def interp_wind(self, rx):
        return self.interp_rx_dat(rx, self.ur_raw) #cm/s

    def interp_rx_dat_log(self, rx, array):
        return 10**(self.interp_rx_dat(rx, np.log10(array)))

    def interp_rx_dat(self, rx, array):
        #Interpolates an array(rx)
        if rx < 1. : return math.nan
        locs = self.rx_raw
        return self.interp(locs, array, rx)

    def interp_frac(self, T, ion):
        #Interpolate the ionization fraction as f(T)
        locs = ion['chTemps']
        func = np.log10(ion['chFracs'])
        temp = np.log10(T)

                #plt.plot(locs, func)
                #plt.xlabel('Temperature')
                #plt.ylabel('Fraction')
                #plt.show()
        return 10**self.interp(locs, func, temp)

    def interp_recomb(self, ion, T):
        #Interpolate the ionization fraction as f(T)
        locs = np.log10(ion['recomb_T'])
        func = np.log10(ion['recomb'])
        temp = np.log10(T)

                #plt.plot(locs, func)
                #plt.xlabel('Temperature')
                #plt.ylabel('Fraction')
                #plt.show()
        return 10**self.interp(locs, func, temp)

    def interp_collis(self, ion, T):
        #Interpolate the ionization fraction as f(T)
        locs = np.log10(ion['collis_T'])
        func = np.log10(ion['collis'])
        temp = np.log10(T)

                #plt.plot(locs, func)
                #plt.xlabel('Temperature')
                #plt.ylabel('Fraction')
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

    def interp_upsilon(self, X, ion):
        #Figures out upsilon as f(X)
        locs = ion['splinedUpsX']
        func = ion['splinedUps']
        return self.interp(locs, func, X)

    def interp(self, X, Y, K):
        #Takes in X and Y and returns linearly interpolated Y at K
        if K >= np.amax(X) or K < np.amin(X) or np.isnan(K): return np.nan
        TInd = int(np.searchsorted(X, K)) - 1
        if TInd + 1 >= len(Y): return np.nan
        val1 = Y[TInd]
        val2 = Y[TInd+1]
        slope = val2 - val1
        step = X[TInd+1] - X[TInd]
        discreteT = X[TInd]
        diff = K - discreteT
        diffstep = diff / step
        return val1 + diffstep*(slope)

    def interp_map(self, map, x, y):
        return map[self.__find_nearest(self.BMap_x, x)][self.__find_nearest(self.BMap_y, y)]

    #def interp_spectrum(self, lamAx):
    #    interp.interp1d()

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

class element:
    def __init__(self, env, ion):
        self.env = env
        self.ion = ion

        
        #Get elemental data
        self.abundance = ion['abundance']
        self.mAtom = ion['mIon']
        self.nIons = ion['element'] + 1
        self.name = ion['ionString']
        
        self.ratList = self.loadRatList() 

        #ionNum = 5
        #tlist = list(np.logspace(3,8,100))
        #ratios = [self.ratio(ionNum, Temp) for Temp in tlist]
        #plt.semilogx(tlist, ratios)
        #plt.show()

        #Make a Height Axis
        self.nRGrid = 2000
        #self.rGrid = np.linspace(1.04, 10, self.nRGrid).tolist()
        
        base = 10
        self.rGrid = np.logspace(np.log(1.04)/np.log(base), np.log(20)/np.log(base), self.nRGrid, base = base).tolist()
        self.zGrid = [r-1 for r in self.rGrid]

        #plt.loglog(self.rGrid, np.ones_like(self.rGrid), 'o')
        #plt.show()

        self.simulate_equilibrium()
        self.simulate_all_NLTE()

        #self.plotTimes()
        self.plotAll()


    def simulate_equilibrium(self):
        """Find the equilibrium ionization balance for every temperature"""

        #Find the N_element as a function of height
        self.rhoGrid = [self.env.interp_rho(rx) for rx in self.rGrid] #g/cm^3
        nTotals = [rho*self.abundance/self.mAtom for rho in self.rhoGrid] #1/cm^3

        #Make a temperature axis as a function of height
        self.tGrid = [self.env.interp_T(rx) for rx in self.rGrid] # # 
        self.tGrid = np.logspace(3,8,self.nRGrid)#

        #Create an empty matrix
        nGrid = np.zeros((self.nIons+1,self.nRGrid))
        normGrid = np.zeros_like(nGrid)

        for temp, nTot, jj in zip(self.tGrid, nTotals, np.arange(self.nRGrid)):
        #For every height point
            
            #Create the mathematical series that determines the number in the ground state.
            series = self.groundSeries(temp)
            
            #Store the total and ground state populations
            nGrid[0][jj] = nTot
            nGrid[1][jj] = nTot/series

            normGrid[0][jj] = nGrid[0][jj] / nTot
            normGrid[1][jj] = nGrid[1][jj] / nTot
          
            #Determine the populations of the higher states
            for ionNum in np.arange(2,self.nIons+1):
                nGrid[ionNum][jj] = nGrid[ionNum-1][jj] * self.recRate(ionNum, temp) / self.colRate(ionNum - 1, temp)
                normGrid[ionNum][jj] = nGrid[ionNum][jj] / nTot
                
        #Store Values
        self.nGrid = nGrid #1/cm^3
        self.normGrid = normGrid


        #rlong = [0] + self.rGrid
        ##np.arange(self.nIons+2),rlong,
        #plt.pcolormesh(np.log(self.normGrid), vmin = -20, vmax = 1)
        #plt.xlabel('Height')
        #plt.colorbar()
        #plt.ylabel('Ion')
        #plt.title('Equilibrium Ionization')
        #plt.show()


    def nRead(self, ionNum, r): 
        return self.env.interp(self.rGrid, self.nGrid[ionNum], r)


    def groundSeries(self, temp):
        series = 1
        thisList = np.arange(len(self.ratList)).tolist()
        for i in self.ratList:
            #For each necessary term in the series
            term = 1
            # print(thisList)
            for ii in thisList:
                #Make that term the multiplication of the remaining ratios
                term *= self.ratio(ii+2, temp)
            # print(term)
            series += term
            (thisList.pop()) #pop the highest valued ratio
        # print('series = {}'.format(series))

        return series


    def simulate_all_NLTE(self):
        """Find the NLTE densities as a function of densfac"""

        doPlot = False
        #Define the grid in the densfac dimension
        self.densPoints = 1
        dMin = 1
        dMax = 3
        densGrid = np.linspace(dMin, dMax, self.densPoints).tolist()

        #Create a multidimensional array to hold everything
        self.bigGrid = np.zeros((self.densPoints, self.nIons+1, self.nRGrid))
        self.bigNormGrid = np.zeros_like(self.bigGrid)

        colors = ['k','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        colorcycler = cycle(colors)

        #Populate the array
        for densfac, dd in zip(densGrid, np.arange(self.densPoints)):
            self.bigGrid[dd], self.bigNormGrid[dd] = self.simulate_one_NLTE(densfac)
            
            doNorm = False

            lw = 2
            if doPlot:
                plt.figure()
                for nn in np.arange(self.nIons+1):
                    clr = next(colorcycler)

                    if doNorm:
                        plt.loglog(self.zGrid, self.bigNormGrid[dd,nn],c=clr, marker='o', lw=lw, label=nn)
                        plt.loglog(self.zGrid, self.normGrid[nn],   c=clr,ls=':', label=nn)
                    else:
                        plt.loglog(self.zGrid, self.bigGrid[dd,nn], marker='o',c=clr,lw=lw, label=nn)
                        plt.loglog(self.zGrid, self.nGrid[nn],   c=clr,ls=':', label=nn)
                    lw=1
                #plt.pcolormesh(np.log(self.bigGrid[dd]), vmin = -20, vmax = 1)
                plt.legend()
                plt.xlabel('Height')
                plt.ylabel('Ion')
                plt.title(self.name)
                #plt.ylim([10**-8,1])
                #plt.title('NLTE Ionization')
                plt.show()




    def simulate_one_NLTE(self, densfac):

        #Parameter for when to start NLTE
        self.LTEstart = 100

        #Pull in the equilibrium calculation and adjust density
        neGrid = copy.deepcopy(self.nGrid * densfac)
        normGrid = copy.deepcopy(self.normGrid)

        #Calculate the wind velocities on the rGrid
        self.uGrid = [self.env.interp_wind(rx)/densfac**0.5 for rx in self.rGrid] #cm/s

        rs2cm = self.env.r_Mm * 10**8

        #Iterate up in r
        for temp, rr, ur, jj in zip(self.tGrid, self.rGrid, self.uGrid, np.arange(self.nRGrid)):
            r = rr * rs2cm #cm
            if jj < 1: continue #Skip the first column

            #For every height point
            dr = (self.rGrid[jj] - self.rGrid[jj-1]) * rs2cm #change in height    #cm
            du = self.uGrid[jj] - self.uGrid[jj-1] #change in wind over that height #cm/s

            rho = self.rhoGrid[jj] #g/cm^3
            nE = rho*0.9/self.env.mP # num/cm^3

            #Get the densities of all ions at the last height
            n = neGrid[:, jj-1] #1/cm^3
            n = np.append(n, 0) #make there be a zero density higher state
            
            for ionNum in np.arange(1,self.nIons+1):
                #For each ion

                ##Should we use NLTE here?
                t_C,t_E = self.particleTimes(ionNum, jj) #s
                timeRatio = t_E/t_C
                #if collisions are happening: do not
                if timeRatio > self.LTEstart: continue
                
                #Ionization and Recombination
                C = n[ionNum-1]*self.colRate(ionNum-1, temp) + n[ionNum+1]*self.recRate(ionNum+1, temp) 
                - n[ionNum] * (self.colRate(ionNum, temp) + self.recRate(ionNum, temp))
                #1/cm^3 * cm^3/s = 1/s

                RHS = 0
                RHS +=  10**0 * nE*C/ur # 1/cm^3 * 1/s * s/cm = 1/cm^4
                RHS -=  2*n[ionNum]/r # 1/cm^3 / cm = 1/cm^4
                RHS -=  n[ionNum]/ur*du/dr # 1/cm^3 * s/cm * cm/s / cm = 1/cm^4

                #RHS =  nE*C/ur - 2*n[ionNum]/r  - n[ionNum]/ur*du/dr

                change = dr * RHS
                neGrid[ionNum, jj] = n[ionNum] + dr * RHS
                normGrid[ionNum, jj] = neGrid[ionNum, jj]/neGrid[0,jj]


        return neGrid, normGrid

    def particleTimes(self,ionNum,jj):
        """Determine the Collision and the Expansion time at a Radius"""
        
        #Aquire relevant parameters
        rho = self.rhoGrid[jj] #g/cm^3
        nE = rho*0.9/self.env.mP # num/cm^3
        T = self.tGrid[jj] #Kelvin
        wind = self.uGrid[jj] #cm/s

        #Do a derivative
        rs_cm = self.env.r_Mm * 10**8
        dr = (self.rGrid[jj] - self.rGrid[jj-1]) * rs_cm #cm
        drho = self.rhoGrid[jj] - self.rhoGrid[jj-1] #g/cm^3

        #Expansion Time
        t_E = np.abs(rho*dr/(wind*drho)) #Expansion Time in s
     

        #Find rates to state below
        recomb1 = self.recRate(ionNum, T) #cm^3/s
        collis1 = self.colRate(ionNum-1, T) #cm^3/s

        #Find rates to state above
        recomb2 = self.recRate(ionNum+1, T) #cm^3/s
        collis2 = self.colRate(ionNum, T) #cm^3/s
        
        #Find times for both rates
        t_C1 = 1/(nE*(recomb1+collis1)) #Collisional Time in s
        t_C2 = 1/(nE*(recomb2+collis2)) #Collisional Time in s

        #Use only the appropriate times
        if ionNum < 2: t_C = t_C2
        elif ionNum >= self.nIons: t_C = t_C1
        else: #What would be the appropriate average to use here?
            #t_C = (t_C1 + t_C2)/2
            t_C = np.sqrt(t_C1*t_C2)

        return t_C, t_E

    def plotTimes(self):
        plt.figure()
        for ionNum in np.arange(1,self.nIons+1):
            col = []
            exp = []
            for jj in np.arange(self.nRGrid):
                t_C,t_E = self.particleTimes(ionNum, jj)
                col.append(t_C)
                exp.append(t_E)
            
            #plt.loglog(self.rGrid, col)
            #plt.loglog(self.rGrid, exp, 'b')
            plt.loglog(self.zGrid, [e/c for c,e in zip(col, exp)])
            plt.ylabel('Seconds')
            plt.xlabel('Solar Radii')
            plt.axhline(10**2, ls=':', c='k')
        plt.show()

    def ratio(self, ionNum, temp):
        """Returns recomb(ionNum) / collis(ionNum - 1)"""
        return self.recRate(ionNum, temp) / self.colRate(ionNum - 1, temp)

    def recRate(self, ionNum, temp):
        """Return R, the recombination from I to I-1"""
        ind = ionNum - 2
        if ind < 0: return 0
        if ind > len(self.ionArray)-1: return 0
        return self.ratList[ind][1](temp) #cm^3/s
 
    def colRate(self, ionNum, temp):
        """Return C, the collision rate from I to I+1"""
        ind = ionNum - 1
        if ind < 0: return 0
        if ind > len(self.ionArray)-1: return 0
        return self.ratList[ind][0](temp) #cm^3/s




    def loadRatList(self):
        '''Load in the recombination and collision rates for all of the ions of this element'''
        ratList = []
        self.ionArray = np.arange(1, self.nIons)

        colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        colorcycler = cycle(colors)

        for ionNum in self.ionArray:
            T_array, thisRecomb = self.env.cLoadOneRecombRate(self.ion['ionString'], ionNum + 1)
            T_array, thisCollis = self.env.cLoadOneCollisRate(self.ion['element'], ionNum)
            
            recombFunc = partial(self.env.interp,T_array,thisRecomb)
            collisFunc = partial(self.env.interp,T_array,thisCollis)
            ratList.append((recombFunc, collisFunc))

            if False:
                #Plot all the recombination and collision times
                clr = next(colorcycler)
                plt.loglog(T_array, thisRecomb, clr + '-', label = 'recomb_{}'.format(ionNum))
                plt.loglog(T_array, thisCollis, clr + '--', label = 'collis_{}'.format(ionNum))
                plt.title('{}_{}'.format(self.ion['ionString'], ionNum))
                plt.legend()
                plt.xlabel('Temperature')
                plt.ylabel('Rate')
        #plt.show()


        return ratList

    def getN(self, ionNum, rx):
        thisState = self.nGrid[ionNum]
        self.env.interp(self.rGrid, thisState, rx)

    def plotAll(self):
        #import pdb; pdb.set_trace()
        #newIonArray = np.arange(self.nIons+2) - 0.5
        # np.append(newIonArray, self.ionArray[-1]+1) 
        # import pdb; pdb.set_trace()
        # plt.pcolormesh(self.rGrid, newIonArray, np.log10(self.nGrid))
        # plt.title('Element: {}'.format(self.ion['ionString']))
        # plt.ylabel('Ionization Stage')
        # plt.xlabel('rx')
        # plt.colorbar()
        # plt.show()
        plt.figure()

        absolute = False
        vsHeight = False

        if absolute:
            plt.title('{}, absolute'.format(self.ion['ionString']))
            grid = self.nGrid
        else:
            plt.title('{}, relative'.format(self.ion['ionString']))
            grid = self.normGrid

        if vsHeight:
            plt.xlabel('Height')
            absiss = self.rGrid
            plt.ylim([1e-8, 2])
        else:
            plt.xlabel('Temperature')
            absiss = self.tGrid
            plt.ylim([1e-3, 2])

        plt.ylabel('Population')
        for ionNum in np.arange(self.nIons+1):
            plt.plot(absiss, (grid[ionNum]), label = ionNum)
        plt.legend()
        
        plt.xscale('log')
        plt.yscale('log')
        plt.show()



####################################################################                            
##                           Simulation                           ##
####################################################################

## Level 0: Simulates physical properties at a given coordinate
class simpoint:
    g_Bmin = 3.8905410
    ID = 0  
    #Level 0: Simulates physical properties at a given coordinate
    def __init__(self, cPos = [0.1,0.1,1.5], grid = None, env = None, findT = True, pbar = None, copyPoint = None):
        #Inputs
        self.didIonList = []
        self.grid = grid
        self.env = env
        self.ions = copy.deepcopy(self.env.getIons(self.env.maxIons))

        if findT is None:
            self.findT = self.grid.findT
        else: self.findT = findT

        self.loadParams(copyPoint)
        
        self.relocate(cPos)

        if pbar is not None:
            pbar.increment()
            pbar.display()

    def relocate(self, cPos, t = 0):
        self.cPos = cPos
        self.pPos = self.cart2sph(self.cPos)
        self.rx = self.r2rx(self.pPos[0])
        self.zx = self.rx - 1
        self.maxInt = 0
        self.totalInt = 0
        #Initialization
        self.findTemp()
        self.findFootB()
        self.__findFluxAngle()
        self.findDensity()
        self.findTwave()
        self.__streamInit()
        self.findSpeeds(t)
        #self.findQt(self.T)
        self.findUrProj()       
        self.findDPB()
        
        return self


    def loadParams(self, copyPoint):
        if copyPoint is None:
            self.useWaves = simpoint.g_useWaves
            self.useWind = simpoint.g_useWind
            self.useFluxAngle = simpoint.g_useFluxAngle
            self.Bmin = simpoint.g_Bmin

            self.bWasOn = self.useB
            self.waveWasOn = self.useWaves
            self.windWasOn = self.useWind
        else:
            self.useWaves = copyPoint.useWaves
            self.useWind = copyPoint.useWind
            self.useFluxAngle = copyPoint.useFluxAngle
            self.Bmin = copyPoint.Bmin

  ## Temperature ######################################################################
    def findTemp(self):
        self.T = self.interp_rx_dat_log(self.env.T_raw) #Lowest Temp: 1200

  ## Magnets ##########################################################################
    def findFootB(self):
        #Find B
        self.__findfoot_Pos()
        x = self.foot_cPos[0]
        y = self.foot_cPos[1]
        if self.useB:
            if self.voroB: self.footB = self.env.interp_map(self.env.BMap, x, y)
            else: self.footB = self.env.interp_map(self.env.BMap_smoothed, x, y)
            #self.footB = self.env.BMap[self.__find_nearest(self.env.BMap_x, x)][self.__find_nearest(self.env.BMap_y, y)]

        else: self.footB = 0

    def __findfoot_Pos(self):
        #Find the footpoint of the field line
        self.f = self.env.getAreaF(self.rx)
        theta0_edge = self.env.theta0 * np.pi/180.
        theta_edge  = np.arccos(1. - self.f + self.f*np.cos(theta0_edge))
        edge_frac   = theta_edge/theta0_edge 
        coLat = self.pPos[1] /edge_frac
        self.foot_pPos = [self.env.rstar+1e-2, coLat, self.pPos[2]]
        self.foot_cPos = self.sph2cart(self.foot_pPos)



    def __findStreamIndex(self):
        x = self.foot_cPos[0]
        y = self.foot_cPos[1]
        self.streamIndex = self.env.interp_map(self.env.label_im, x, y)
        #self.streamIndex = self.env.label_im[self.__find_nearest(self.env.BMap_x, x)][self.__find_nearest(self.env.BMap_y, y)]

  ## Density ##########################################################################
    def findDensity(self):
        #Find the densities of the grid point
        self.densfac = self.__findDensFac()
        self.rho = self.__findRho(self.densfac) #Total density
        self.nE = 0.9*self.rho/self.env.mP #electron number density

        for ion in self.ions:
            ion['frac'] = self.env.findIonFrac(ion, self.T, self.rho) if self.useIonFrac else 1 #ion fraction
            ion['N'] = np.abs(0.8 * ion['frac'] * ion['abundance'] * self.rho/self.env.mP) #ion number density 

    def __findDensFac(self):
        # Find the density factor
        Bmin = self.Bmin
        Bmax = 50

        if self.footB < Bmin: self.B0 = Bmin
        elif self.footB > Bmax: self.B0 = Bmax
        else: self.B0 = self.footB
        dinfty = (self.B0/15)**1.18

        if False: #Plot the density factor lines
            xx = np.linspace(0,3,100)
            dd = np.linspace(0,20,5)
            for d in dd:
                plt.plot(xx + 1, self.__densFunc(d, xx), label = "{} G".format(d))
            plt.xlabel('r/$R_\odot$')
            plt.ylabel('$D_r$')
            plt.legend()
            plt.show()

        if self.useB:
            return self.__densFunc(dinfty, self.zx)
        else: return 1

    def __densFunc(self,d,x):
        return (0.3334 + (d - 0.3334)*0.5*(1. + np.tanh((x - 0.37)/0.26)))*3

    def __findRho(self, densfac): 
        return self.env.interp_rho(self.rx) * densfac
        #return self.interp_rx_dat_log(self.env.rho_raw) * densfac  

  ## pB stuff ##########################################################################

    def findDPBold(self):
        imp = self.cPos[2]
        r = self.pPos[0]
        u = 0.63
        R_sun = 1
        sigmaT = 5e-12#6.65E-25 #

        eta = np.arcsin(R_sun/r)
        mu = np.cos(eta)

        f = mu**2 / np.sqrt(1-mu**2) * np.log(mu/(1+np.sqrt(1-mu**2)))
        A = mu - mu**3
        B = 1/4 - (3/8)*mu**2 + f/2 *( (3/4)*mu**2 - 1 )


        self.dPB = 3/16* sigmaT * self.nE * (imp / r)**2 * ( (1-u)*A + u*B ) / (1-u/3)


        return self.dPB

    def findDPB(self):
        imp = self.cPos[2]
        x = self.cPos[0]
        r = self.pPos[0]
        u = 0.63
        R_sun = 1
        sigmaT = 7.95E-26 #cm^2
        I0 = 2.49E10 #erg/ (cm^2 s sr)

        Tau = np.arctan2(x, r)
        Omega = np.arcsin(R_sun/r)

        A = np.cos(Omega)*np.sin(Omega)**2

        B = -1/8* ( 1 - 3*np.sin(Omega)**2 - np.cos(Omega)**2/np.sin(Omega) * (1+3*np.sin(Omega)**2) * np.log( (1+np.sin(Omega)) / np.cos(Omega) ) )

        self.dPB = 0.5*np.pi*sigmaT*I0 * self.nE * np.cos(Tau)**2 * ( (1-u)*A + u*B )




        return self.dPB


  ## Radiative Transfer ################################################################

    def getProfiles(self, lenCm = 1):
        self.lenCm = lenCm
        profiles = []
        for ion in self.ions:
            profileC = self.collisionalProfile(ion)
            profileR = self.resonantProfile(ion)
            profiles.append([profileC,profileR])
        self.env.plotMore = False
        return profiles

    def collisionalProfile(self, ion):
        """Generate the collisionally excited line profile"""
        lam0 = ion['lam00'] #Angstroms
        vTh = np.sqrt(2 * self.env.KB * self.T / ion['mIon']) #cm/s
        deltaLam = lam0  * vTh / self.env.c #ang
        lamLos =  lam0 * self.vLOS / self.env.c #ang

        expblock = (ion['lamAx'] - lam0 - lamLos)/(deltaLam) #unitless

        lamPhi = 1/(deltaLam * self.env.rtPi) * np.exp(-expblock*expblock) #1/ang

        if self.useIonFrac: nn = ion['N'] #/cm^3
        else: nn = self.nE
        
        const = self.env.hergs * self.env.c / self.env.ang2cm(lam0) / 4 / np.pi #ergs s 

        profileC = self.lenCm * const * self.nE * nn * self.env.findQt(ion, self.T) * lamPhi # ergs /s /sr /cm^2 /ang ;  Q=[cm^3/s^2/sr]

        ion['profileC'] = profileC #ergs / (s cm^2 sr angstrom)
        ion['totalIntC'] = np.sum(profileC)
        ion['Cmax'] = np.max(profileC)

        return profileC


    def resonantProfile(self, ion):
        """Generate the resonantly scattered line profile"""

        ## Determine the Relevant Velocities #######

        #Thermal Velocity
        Vth = np.sqrt(2 * self.env.KB * self.T / ion['mIon']) #cm/s
        inv_Vth = 1/Vth
        
        #Normed LOS and radial velocities
        los_Vth = self.vLOS*inv_Vth
        radialV = self.ur 
        rad_Vth = radialV*inv_Vth
       

        ## Create the prime (incident) axes ########

        #Find the necessary limits for this redshift
        lamShift =  ion['lam00'] * radialV / self.env.c #Redshift of this point
        lamCenter = ion['lam00'] - lamShift #Center of this points spectrum
        deltaLam =  ion['lam00'] * Vth / self.env.c #Width of the line here
        
        sigma = 4
        lowLam = lamCenter - sigma*deltaLam #New limits for this point
        highLam = lamCenter + sigma*deltaLam #New limits for this point
        throw = highLam - lowLam #total width in angstroms

        ## Create the Incident array ##############
        
        #Interpolate it
        rezPerAng = 150
        minRez = 60
        longRez = int(throw * rezPerAng)
        rez = np.max((longRez,minRez))

        #rez = 125

        #if self.env.plotMore and not ion['ionString'] in self.didIonList: 
        #    self.didIonList.append(ion['ionString'])
        #    print("E: {}, throw = {:.3}, rez = {}".format(ion['ionString'], throw, rez))

        lamAxPrime = np.linspace(lowLam, highLam, rez)
        I0array = ion['I0interp'](lamAxPrime) ##ergs/cm^2/sr/Angstrom
        #lamAxPrime = self.env.cm2ang(lamAxPrime)

        #Just use the whole thing
        #lamAxPrime, I0array = ion['lamAxPrime'], ion['I0array'] ##ergs/cm^2/sr/Angstrom

        #Just pull the important part of the raw spectrum at original resolution
        #lamAxPrime, I0array = self.env.returnSolarSpecLamFast(lowLam, highLam, ion['lamAxPrime'], ion['I0array'])
            #angstroms, ergs/s/cm^2/sr/Angstrom

        #Convert from lam to nu
        nuAxPrime = self.env.lamAx2nuAx(lamAxPrime) #Hz
        nuI0Array = self.env.lam2nuI0(I0array, lamAxPrime) #ergs/s/cm^2/sr/Hz
        dNu = np.abs(np.mean(np.diff(nuAxPrime))) #Hz


        ## Calculate Scalar Prefactors #############
        #Geometric factors
        ro = np.sqrt(self.cPos[0]*self.cPos[0] + self.cPos[1]*self.cPos[1])
        Theta = np.arccos(ro/self.pPos[0])
        alpha = np.cos(Theta)
        beta = np.sin(Theta) 
        inv_beta = 1/beta

        #Other scalar calculations
        nu0 = ion['nu0']
        if self.useIonFrac: Nion = ion['N']
        else: Nion = self.nE # num/cm^3    
        deltaNu = nu0 * Vth / self.env.c #1/s
        scalarFactor = self.lenCm * self.env.hergs * nu0 / (4*np.pi) * ion['B12'] * Nion * dNu * self.findSunSolidAngle() #  hz*hz 

        #g = ion['E'] + 3*(1-ion['E'])*alpha*alpha #phase function
        g1 = (1-ion['E1']/4) + 3/4 *ion['E1']*alpha*alpha #new phase function

        Rfactor = g1/(np.pi * beta * deltaNu*deltaNu) # 1/(hz*hz)
        preFactor = scalarFactor * Rfactor # unitless

        ## The main matrix kernal ##################
        #Create a column and a row vector
        zeta =      (ion['nuAx'] -nu0) / deltaNu - los_Vth
        zetaPrime = (nuAxPrime   -nu0) / deltaNu - rad_Vth

        zetaTall = zeta[np.newaxis].T
        zetaDiffBlock = (zetaTall-alpha*zetaPrime)*inv_beta

        #Use the vectors to create a matrix
        R = np.exp(-zetaPrime*zetaPrime)*np.exp(-zetaDiffBlock*zetaDiffBlock) #unitless

        #Apply that matrix to the incident light profile
        profileRnu = preFactor * np.dot(R, nuI0Array) #ergs/s/cm^2/sr/Hz

        #Convert from nu back to lam
        profileR = profileRnu * ion['nuAx'] / ion['lamAx'] #ergs/s/cm^2/sr/Angstrom

        ## Store values
        ion['profileR'] = profileR
        ion['totalIntR'] = np.sum(ion['profileR'])

        ## Plot Stuff # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
        #Plot the slice being used as incident light
        stepdown = 0.2
        def gauss_function(x, a, x0, sigma, b): return a*np.exp(-(x-x0)**2/(2*sigma**2)) + b

        plotIncident = False 

        if plotIncident and self.pPos[0] + stepdown < self.env.lastPos and ion['ionString'] == 's':
            self.env.lastPos = self.pPos[0]
            
            plt.figure()
            plt.axvline(lowLam)
            plt.axvline(highLam)
            plt.axvline(lamCenter)
            plt.title("{}_{}".format(ion['ionString'], ion['ion']))
            plt.plot(ion['lamAxPrime'], ion['I0array'], label="Full")
            
            plt.plot(ion['lamAx'], gauss_function(ion['lamAx'], 50, lamCenter, deltaLam, 0))

            plt.plot(lamAxPrime, I0array, '.-', label = "Slice")
            ionName= "{}_{}".format(ion['ionString'], ion['ion'])
            plt.title("{}\nR = {:.3}, X,Z = {:.3}, {:.3}\n radialV = {:.4}".format(ionName, self.pPos[0], self.cPos[0], self.cPos[2], self.env.cm2km(radialV)))
            plt.legend()

            if False:
                plt.show()
            else:
                savePath = "../fig/2018/steps"
                plt.savefig("{}/{:.3}.png".format(savePath,self.pPos[0]))
                plt.close()

        #Plot the Generated Profile
        #if ion['ionString'] == 's':
        #plt.plot(lamAx, profileR, '.-')
        #plt.show() #leave this blank to see them all on the same graph

        #lastTime = time.time()      
        #thisTime =  time.time()
        #elapsed = thisTime-lastTime
        #print('time: {:.3}, e: {}, rez: {}, rat: {:.3}'.format(elapsed*1000, ion['ionString'], rez, elapsed*100000 / rez))


        return ion['profileR']


    def findSunSolidAngle(self):
        """Return the proportion of the sky covered by the sun"""
        return 0.5*(1-np.sqrt(1-(1/self.rx)**2))

  ## Velocity ##########################################################################

    def findSpeeds(self, t = 0):
        #Find all of the static velocities
        
        if self.useWind: self.uw = self.__findUw()
        else: self.uw = 0
        self.vAlf = self.__findAlf()
        self.vPh = self.__findVPh()
        if self.useWaves: self.vRms = self.__findvRms()
        else: self.vRms = 0

        self.findWaveSpeeds(t)

    def findWaveSpeeds(self, t = 0):
        #Find all of the wave velocities

        if self.wavesVsR and self.useWaves: 
            offsetTime = 410.48758
            #Set the wave time
            self.t1 = t + self.alfT1
            #self.t2 = t + self.alfT2
            #Get the wave profile at that time
            self.alfU1 = self.interp_R_wave(self.t1)
            #self.alfU2 = self.interp_R_wave(self.t2)
            #Match the boundary of Braid with the time method
            if self.matchExtrap:
                self.alfU1 = self.alfU1 if not np.isnan(self.alfU1) else [np.nan if self.zx < 0 else self.vRms*self.xi1(self.t1- self.twave + offsetTime)].pop()
                #self.alfU2 = self.alfU2 if not np.isnan(self.alfU2) else [np.nan if self.zx < 0 else self.vRms*self.xi2(self.t2- self.twave)].pop()
            else:
                self.alfU1 = self.alfU1 if not np.isnan(self.alfU1) else [np.nan if self.zx < 0 else 0].pop() 
                #self.alfU2 = self.alfU2 if not np.isnan(self.alfU2) else [np.nan if self.zx < 0 else 0].pop()
            #Modify the waves based on density
            self.alfU1 = self.alfU1/self.densfac**0.25
            #self.alfU2 = self.alfU2/self.densfac**0.25

        elif self.useWaves:
            #Just the time method
            self.t1 = t - self.twave + self.alfT1
            self.t2 = t - self.twave + self.alfT2
            self.alfU1 = self.vRms*self.xi1(self.t1) /self.densfac**0.25
            #self.alfU2 = self.vRms*self.xi2(self.t2) /self.densfac**0.25
        else: self.alfU1 = 0
        #Rotate the waves and place them in correct coordinates
        uTheta = self.alfU1 #* np.sin(self.alfAngle) + self.alfU2 * np.cos(self.alfAngle)
        uPhi =   self.alfU1 #* np.cos(self.alfAngle) - self.alfU2 * np.sin(self.alfAngle)
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

    def interp_R_wave(self, t):
        t_int = int(t%self.env.R_ntime)
        R_Array = self.env.R_vlos[:,t_int]
        V = self.env.interp(self.env.R_zx, R_Array, self.zx)
        return V * 1e5

    def __findFluxAngle(self):
        dr = 1e-4
        r1 = self.rx
        r2 = r1 + dr

        thetar1 = np.arccos(1 - self.env.getAreaF(r1)*(1-np.cos(self.foot_pPos[1])))
        thetar2 = np.arccos(1 - self.env.getAreaF(r2)*(1-np.cos(self.foot_pPos[1])))
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

        uw = self.__findUw()
        vRms = self.__findvRms()

        self.urProj =  (np.sin(theta + delta) *  uw )**2 * self.rho**2
        self.rmsProj = (np.cos(theta + delta) * vRms)**2 * self.rho**2
        self.temProj =                            self.T * self.rho**2

    def __streamInit(self):
        self.__findStreamIndex()
        self.env.streamRand.seed(int(self.streamIndex))
        thisRand = self.env.streamRand.random_sample(3)

        if self.wavesVsR:
            last1 = self.env.R_ntime
            last2 = last1
        else:
            last1 = self.env.last_xi1_t
            last2 = last1

        self.alfT1 =  int(thisRand[0] * last1)
        self.alfT2 =  int(thisRand[1] * last2)
        self.alfAngle = thisRand[2] * 2 * np.pi


    def __findUw(self):
        #Wind Velocity
        #return (1.7798e13) * self.fmax / (self.num_den * self.rx * self.rx * self.f)
        return self.env.interp_wind(self.rx) / self.densfac**0.5
        #return self.interp_rx_dat(self.env.ur_raw) / self.densfac**0.5

    def __findAlf(self):
        #Alfven Velocity
        #return 10.0 / (self.f * self.rx * self.rx * np.sqrt(4.*np.pi*self.rho))
        return self.interp_rx_dat(self.env.vAlf_raw) / np.sqrt(self.densfac)

    def __findVPh(self):
        #Phase Velocity
        return self.vAlf + self.uw 
    
    def __findvRms(self):
        return self.env.findvRms(self.rx)



    def __findVLOS(self, nGrad = None):
        if nGrad is not None: self.nGrad = nGrad
        else: self.nGrad = self.grid.ngrad
        self.vLOS = np.dot(self.nGrad, self.cU)
        #self.vLOS = self.alfU1 #FOR TESTING
        return self.vLOS

    def __findVLOS2(self, vel, nGrad = None):
        if nGrad is None: nGrad = self.grid.ngrad
        vLOS2 = np.dot(nGrad, vel)
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
        
        if self.findT and not self.wavesVsR:
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

    def __find_nearest(self,array,value):
        #Returns the index of the point most similar to a given value
        array = array-value
        np.abs(array, out = array)
        return array.argmin()


    def interp_rx_dat(self, array):
        #Interpolates an array(rx)
        if self.rx < self.env.rx_raw[0] : return math.nan
        rxInd = np.int(np.searchsorted(self.env.rx_raw, self.rx) -1)
        val1 = array[rxInd]
        val2 = array[rxInd+1]
        slope = val2 - val1
        step = self.env.rx_raw[rxInd+1] - self.env.rx_raw[rxInd]
        discreteRx = self.env.rx_raw[rxInd]
        diff = self.rx - discreteRx
        diffstep = diff / step
        return val1+ diffstep*(slope)

    def interp_rx_dat_log(self, array):
        return 10**(self.interp_rx_dat(np.log10(array)))

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
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.root = self.rank == 0
        self.size = self.comm.Get_size()

        self.inFindT = findT
        self.inIL = iL
        self.inN = N

        self.grid = gridObj
        self.print = printOut
        self.findT = findT
        self.env = envObj
        self.getProf = getProf
        self.timeAx = timeAx
        self.randomizeTime()  

        self.ions = self.env.getIons(self.env.maxIons)

        self.throwCm = self.grid.norm * self.env.r_Mm * 1e8 #in cm
        self.sims = []
        self.repoint(gridObj)


    def simulate_now(self):
  
        if type(self.grid) is grid.plane: 
            doBar = True
            self.adapt = False
            self.Npoints = self.grid.Npoints
        else: doBar = False
        
        if doBar and self.print: bar = pb.ProgressBar(self.Npoints)
        self.oldSims = self.sims
        self.sims = []

        self.steps = []
        self.pData = []
        #if self.print: print("\nBeginning Simulation...")

        t = time.time()
        stepInd = 0
        rhoSum = 0
        tol = 2500

        for cPos, step in self.grid: 

            if self.oldSims:
                thisPoint = self.oldSims.pop().relocate(cPos)
            else: 
                thisPoint = simpoint(cPos, self.grid, self.env, self.findT) 
                
            if self.adapt:
                #Adaptive Mesh
                thisDens = thisPoint.dPB

                if (thisDens > tol) and self.grid.backflag:
                    self.grid.back()
                    self.grid.set2minStep()
                    self.grid.backflag = False
                    continue
                if thisDens <= tol:
                    self.grid.incStep(1.5)
                    self.grid.backflag = True

            stepInd += 1
            
            self.sims.append(copy.copy(thisPoint))
            self.steps.append(step)
            self.pData.append(thisPoint.Vars())
            
            if doBar and self.print: 
                bar.increment()
                bar.display()
        self.cumSteps = np.cumsum(self.steps)
        if doBar and self.print: bar.display(force = True)

        self.simFinish()

    def simFinish(self):
        self.Npoints = len(self.sims)
        if type(self.grid) is grid.sightline:
            self.shape = [self.Npoints, 1] 
        else: 
            self.shape = self.grid.shape
        self.shape2 = [self.shape[0], self.shape[1], -1]

    def repoint(self, gridObj):
        self.grid  = gridObj
        try: self.index = gridObj.index
        except: self.index = (0,0,0)

        if self.inFindT is None: self.findT = self.grid.findT 
        else: self.findT = self.inFindT
        #print(self.findT)

        if self.inIL is None: self.iL = self.grid.iL
        else: self.iL = self.inIL

        if self.inN is None: self.N = self.grid.default_N
        else: self.N = self.inN 

        if type(self.N) is list or type(self.N) is tuple: 
            self.adapt = True
            self.grid.setMaxStep(1/self.N[0])
            self.grid.setMinStep(1/self.N[1])
        else: 
            self.adapt = False
            self.grid.setN(self.N)


        self.profile = None

        self.simulate_now()
        if self.getProf: self.lineProfile()

        return self

    def refreshData(self):
        self.pData = []
        for thisPoint in self.sims:
            self.pData.append(thisPoint.Vars())

    def get(self, myProperty, dim = None, scaling = 'None', scale = 10, ion = None, refresh = False):
        if refresh: self.refreshData()
        if ion is None:
            propp = np.array([x[myProperty] for x in self.pData])
        else:
            try:propp = np.array([x['ions'][ion][myProperty] for x in self.pData])
            except:propp = np.array([x[myProperty] for x in self.pData])
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
        elif scaling.lower() == 'norm':
            scaleProp = prop/np.amax(prop)
        else: 
            print('Bad Scaling - None Used')
            scaleProp = prop
        return scaleProp

    def plot(self, property, dim = None, scaling = 'None', scale = 10, ion = None, cmap = 'viridis', norm = False, threeD = False, useCoords = True,
             axes = False, center = False, linestyle = 'b', abscissa = None, absdim = 0, yscale = 'linear', xscale = 'linear', extend = 'neither', sun = False, block=True,
             maximize = False, frame = False, ylim = None, xlim = None, show = True, refresh = False, useax = False, clabel = None, suptitle = None, savename = None, nanZero = True, **kwargs):
        unq = ""
        #Create the Figure and Axis
        if useax: 
            ax = useax
            self.fig = ax.figure
        elif frame:
            self.fig, ax = self.grid.plot(iL = self.iL)
        elif threeD:
            self.fig = plt.figure()
            ax = self.fig.add_subplot(111, projection='3d')
        else:
            self.fig, ax = plt.subplots()

        if savename is not None: self.fig.canvas.set_window_title(str(savename))

        if suptitle is not None:
            self.fig.suptitle(suptitle)

        #Get the abscissa
        if abscissa is not None: absc = self.get(abscissa, absdim)
        else: absc = self.cumSteps

        #Condition the Inputs into lists
        if not isinstance(property, (list, tuple)): properties = [property]
        else: properties = property
        if not isinstance(dim, (list, tuple)): dims = [dim]
        else: dims = dim
        if not isinstance(scaling, (list, tuple)): scalings = [scaling]
        else: scalings = scaling
        if not isinstance(scale, (list, tuple)): scales = [scale]
        else: scales = scale

        #make sure they have the right lengths
        multiPlot = len(properties) > 1
        while len(dims) < len(properties): dims.append(None)
        while len(scalings) < len(properties): scalings.append('None')
        while len(scales) < len(properties): scales.append(1)

        #Condition the ion list
        if ion is None: useIons = [None]
        elif ion == -1: useIons = np.arange(len(self.ions))
        elif isinstance(ion, (tuple, list)): useIons = ion
        else: useIons = [ion]

        #Prepare the Plot Styles
        
        lines = ["-","--","-.",":"]
        linecycler = cycle(lines)
        colors = ['C0','C1','C2','C3','C4','C5','C6','C7','C8','C9']
        colorcycler = cycle(colors)

        #Make the plotlist
        plotList = []
        plotLabels = []
        styles = []

        for ii in useIons: #For each ion
            clr = next(colorcycler)
            linecycler = cycle(lines)
            for property, dim, scaling, scale in zip(properties, dims, scalings, scales): #For each property
                ls = next(linecycler)

                #Get the data
                thisProp = np.ma.masked_invalid(self.get(property, dim, scaling, scale, ii, refresh))
                if nanZero: thisProp[thisProp == 0] = np.ma.masked
                if norm: thisProp = thisProp / np.max(thisProp)

                #Append to lists
                plotList.append(thisProp)
                labString = property 
                if ii is not None: labString += ', {}_{}'.format(self.ions[ii]['ionString'], self.ions[ii]['ion'])
                plotLabels.append(labString)
                styles.append(clr+ls)

        if type(self.grid) is grid.sightline:

            #Line Plot
            for prop, lab, style in zip(plotList, plotLabels, styles):
                im = ax.plot(absc, prop, style, label = lab, **kwargs)

            #Label Everything
            ax.legend()
            ax.set_xlabel(abscissa)
            if not multiPlot: ax.set_ylabel(property)
            ax.set_yscale(yscale)
            ax.set_xscale(xscale)
            if ylim is not None: ax.set_ylim(ylim)
            if xlim is not None: ax.set_xlim(xlim)
            if axes:
                ax.axhline(0, color = 'k')
                if abscissa is None:
                    ax.axvline(0.5, color = 'k')
                else: ax.axvline(0, color = 'k')
                #ax.set_xlim([0,1])


        elif type(self.grid) is grid.plane:

            #Unpack the property
            scaleProp = plotList.pop()

            #Get the Coordinates of the points
            xx = self.get('cPos', 0)
            yy = self.get('cPos', 1)
            zz = self.get('cPos', 2)
            coords = [xx,yy,zz]

            #Find the throw in each dim
            xs = np.ptp(xx)
            ys = np.ptp(yy)
            zs = np.ptp(zz)
            diffs = [xs,ys,zs]

            #Find the small dimension and keep the others
            minarg = np.argmin(diffs)

            inds = [0,1,2]
            inds.remove(minarg)
            vind = inds.pop()
            hind = inds.pop()

            nind = 3 - vind - hind
            labels = ['X','Y','Z']

            hcoords = coords[hind]
            vcoords = coords[vind]
            ncoords = coords[nind]
            ax.set_xlabel(labels[hind])
            ax.set_ylabel(labels[vind])

            unq = np.unique(ncoords)
            if len(unq) > 1: otherax = "Non Constant"
            else: otherax = unq

            #Center the Color Scale
            if center:
                vmax = np.nanmax(np.abs(scaleProp))
                vmin = -vmax
            else:
                vmin, vmax = None, None

            if threeD:

                #Precondition
                xmin = np.min(np.min(scaleProp))
                xmax = np.max(np.max(scaleProp))
                newScale = (scaleProp - xmin)/(xmax-xmin)
                mapfunc = getattr(plt.cm, cmap)

                #Plot
                im = ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, facecolors=mapfunc(newScale), shade=False)
                
                #Set plot limits
                xlm = ax.get_xlim()
                ylm = ax.get_ylim()
                zlm = ax.get_zlim()
                lims = [xlm, ylm, zlm]
                diffs = [np.abs(x[0] - x[1]) for x in lims]
                val, idx = max((val, idx) for (idx, val) in enumerate(diffs))
                goodlim = lims[idx]

                if nind == 1: ax.set_xlim(goodlim)
                if nind == 2: ax.set_ylim(goodlim)
                if nind == 3: ax.set_zlim(goodlim)

                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

            else:

                hrez = np.abs(hcoords[1] - hcoords[0])/2
                vrez = np.abs(vcoords[1] - vcoords[0])/2
                if useCoords:
                    im = ax.pcolormesh(hcoords-hrez,vcoords-vrez, scaleProp, cmap=cmap, vmin = vmin, vmax = vmax, **kwargs)
                else:
                    im = ax.imshow(scaleProp, cmap=cmap, vmin=vmin, vmax=vmax, **kwargs)
                ax.set_aspect('equal')
                ax.patch.set(hatch='x', edgecolor='lightgrey')

                if axes: 
                    ax.axhline(0, color = 'k')
                    ax.axvline(0, color = 'k') #Plot axis lines

                try: 
                    if frame:
                        self.fig.subplots_adjust(right=0.89)
                        cbar_ax = self.fig.add_axes([0.91, 0.10, 0.03, 0.8], autoscaley_on = True)
                        self.fig.colorbar(im, cax=cbar_ax, label = clabel, extend = extend)
                    else: self.fig.colorbar(im, label = clabel, extend = extend) #Put the colorbar
                except:pass

            if sun:
                if threeD:
                    self.grid.plotSphere(ax, False)
                else:
                    sunCirc = plt.Circle((0,0),1, facecolor='orange', edgecolor ='k')
                    ax.add_artist(sunCirc)

        else: 
            print("Invalid Grid")
            return


        #Setting the Title
        if multiPlot: property = 'MultiPlot'

        zz =np.unique(self.get('cPos', 2))

        if len(zz) == 1: property = property + ", Z = {}".format(zz[0])

        if ion is None: ionstring = ''
        elif ion == -1: ionstring = 'All Ions'
        elif isinstance(ion, (list, tuple)): ionstring = 'MultiIon'
        elif isinstance(ion, int): 
            i = self.ions[ion]
            ionstring = '{}_{} : {} -> {}, $\lambda_0$: {} $\AA$'.format(i['ionString'], i['ion'], i['upper'], i['lower'], i['lam0'])
        else: assert False

        try:
            nmean = np.mean(ncoords)
            if np.unique(ncoords).size == 1:
                nval = nmean
            else:
                nval = "Varying"
        except: nval = "fail"

        if dim is None:
            try: ax.set_title(property + ", scaling = " + scaling + ', {}={}, {}'.format(labels[nind], unq[0],ionstring))
            except: ax.set_title("{}, scaling = {}".format(property, scaling))
        else:
            ax.set_title(property + ", dim = " + dim.__str__() + ", scaling = " + scaling )

        #Finishing Moves
        if maximize and show: grid.maximizePlot()
        if show: plt.show(block)
        return ax
        

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

    def show(self, short = False):
        #Print all properties and values
        myVars = vars(self.sims[0])
        print("\nSimpoint Properties")
        for ii in sorted(myVars.keys()):
            var = myVars[ii]
           
            if not short: print(ii, " : ", var)
            if short:
                p = False
                try: 
                    l = len(var)
                    if l < 10: p = True
                except: p = True
                if p: print(ii, " : ", var)
                
    def Vars(self):
        #Returns the vars of the simpoints
        return self.sims[0].Vars()

    def Keys(self):
        #Returns the keys of the simpoints
        return self.sims[0].Vars().keys()

    ####################################################################

    #def getProfile(self):
    #    return self.lineProfile()

    def plotProfile(self, which = 'R', norm = False):

        prof = 'profile' + which

        try: self.ions[0][prof]
        except: self.lineProfile()

        for ion in self.ions:
            if norm: pl = ion[prof]/np.max(ion[prof])
            else: pl = ion[prof]
            plt.plot(pl)

        plt.title("Line Profile")
        plt.ylabel("Intensity")
        plt.xlabel("Wavelenght (A)")
        #plt.yscale('log')
        plt.show()

    def randomizeTime(self):
        self.env.timeRand.seed(int(self.env.primeSeed + self.rank))
        self.rOffset = self.env.timeRand.randint(self.env.tmax)
        if self.randTime:
            self.timeAx = [ t + self.rOffset for t in self.timeAx]
            

    def lineProfile(self):

        if not hasattr(self, 'profilesC'):
            
            #Get a line profile integrated over time
            profilesC = []
            profilesR = []
            for ion in self.ions:
                #Create an empty box for the profiles to be put in
                profilesC.append(np.zeros_like(ion['lamAx']))
                profilesR.append(np.zeros_like(ion['lamAx']))

            #Initialize plot stuff
            plotIon = self.ions[-1]
            plotLam = plotIon['lamAx']-plotIon['lam00']
            if self.plotSimProfs: fig, axarray = plt.subplots(nrows=2, ncols=1, sharex=True, figsize = (4.1,6), dpi = 200)
        

            #Initialize values
            ppC = []
            ppR = []
            scale = 4
            urProj = 0
            rmsProj = 0
            temProj = 0
            rho2 = 0
            pB = 0

        
        
            if self.print and self.root: 
                print('\nGenerating Light...')
                bar = pb.ProgressBar(len(self.sims) * len(self.timeAx))

            for point, step in zip(self.sims, self.steps):

                lenCm = step * self.throwCm
                #For each simpoint
                for tt in self.timeAx:
                    #For each time
                    point.setTime(tt)
                    for profileC, profileR, (newProfC, newProfR) in zip(profilesC, profilesR, point.getProfiles(lenCm)):  
                        #For each Ion

                        #Sum the light
                        profC = newProfC #step part moved into lenCm
                        profR = newProfR #step part moved into lenCm
                        profileC += profC 
                        profileR += profR

                    if self.plotSimProfs: 
                        #Some plot things
                        bigprofC = profC*10**scale  
                        bigprofR = profR*10**scale
                        ppC.append(bigprofC)
                        ppR.append(bigprofR)

                    #Sum the ion independent things
                    pB += point.dPB * step
                    urProj += point.urProj * step
                    rmsProj += point.rmsProj * step
                    temProj += point.temProj * step
                    rho2 += point.rho**2 * step

                    if self.print and self.root:
                        bar.increment()
                        bar.display()

                    if self.plotSimProfs: axarray[1].plot(plotLam, bigprofC)


            if self.plotSimProfs:
                axarray[1].plot(plotLam, profileC * 10**scale, 'k')
                axarray[0].plot(plotLam, profileC * 10**scale, 'k')
                axarray[0].stackplot(plotLam, np.asarray(ppC))

                axarray[1].set_yscale('log')
                axarray[1].set_ylim([1e-9* 10**scale, 5e-4* 10**scale])
                axarray[0].set_ylim([0,2.5])
                axarray[1].set_xlim([1030-plotion['lam00'],1033.9-plotion['lam00']])
                fig.text(0.005, 0.5, 'Intensity (Arb. Units)', va='center', rotation='vertical')
                #axarray[0].set_ylabel('Intensity (Arb. Units)')
                #axarray[1].set_ylabel('Intensity (Arb. Units)')
                axarray[1].set_xlabel('Wavelength ($\AA$)')

                #grid.maximizePlot()
                #plt.suptitle('Contributions to a single profile')
                plt.show()


            for ion, profileC, profileR in zip(self.ions, profilesC, profilesR):
                #Store the summed profiles for each ion
                ion['profileC'] = profileC
                ion['profileR'] = profileR

            self.profilesC = profilesC
            self.profilesR = profilesR
            self.pB = pB
            self.urProj = np.sqrt(urProj/rho2)
            self.rmsProj = np.sqrt(rmsProj/rho2)
            self.temProj = (temProj/rho2)

            for point in self.sims: #Not sure how to normalize this
                point.urProjRel = np.sqrt(point.urProj / urProj / rho2)
                point.rmsProjRel = np.sqrt(point.rmsProj / rmsProj / rho2)

            #plt.plot(profile)
            #plt.show()
            if self.print and self.root: bar.display(True)
        return self.profilesC, self.profilesR



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
            self.timeAx = [tt]
            self.lineArray[timeInd][:] = self.lineProfile()[0]
            bar.increment()
            bar.display()  
            timeInd += 1       
        bar.display(force = True)

        self.lineList = self.lineArray.tolist()

        self.timeStats()
        self.slidingWindow()
        self.windowPlot()
        #self.__fitGaussians_t()
        #self.__findMoments_t()
        #self.plotLineArray_t()

    def timeStats(self):
        self.mybatch = batch('noRand').loadBatch()
        self.mybatch.makePSF()
        self.mybatch.II=0
        self.mybatch.impact = self.sims[0].cPos[2]

        #bar = pb.ProgressBar(len(self.times))
        #for profile in self.lineList:
        #    p = self.mybatch.findProfileStats(profile)
        #    self.centroid.append(p[1])
        #    self.sigma.append(p[2])
        #    bar.increment()
        #    bar.display()
        #bar.display(force = True)  

    def slidingWindow(self):

        profileNorm = True
        logbreak = 10

        #Get the Arrays
        profiles = self.lineArray

        #Calculate bounds
        NT = len(self.times)
        longestWindow = NT // 1.5

        self.firstIndex = 0 #longestWindow + 1
        self.lastIndex = NT - 1
        
        theBreak = min(longestWindow/logbreak, 100)

        linpart = np.arange(theBreak)
        logpart = np.logspace( np.log10(theBreak) , np.log10(longestWindow) , 100)
        self.winAx = np.concatenate((linpart, logpart)).astype(int)
        self.nWin = len(self.winAx)
        self.nLam = len(profiles[0][:])

        
        #Generate SlideList/SlideArray
        slideProfiles = []
        slideCentroids = []
        slideSigmas = []

        print("Windowing...")
        bar = pb.ProgressBar(len(self.times))
        bar.display()

        for tt in self.times:
            #For every timestep

            #Get the window
            shiftedProfiles = np.flipud(self.getMaxWindow(profiles, tt, longestWindow, NT))

            #Initialize variables
            profile = np.zeros(self.nLam)
            winArray = np.zeros((self.nWin, self.nLam))
            TT = 0
            ww = 0
            centList = []
            sigList = []

            for window in self.winAx:
                #Sum profiles until current window length
                while window >= TT:
                    profile += shiftedProfiles[TT]
                    TT+=1

                #profile = self.getWindow(profiles,tt,window,NT)

                if profileNorm: outProf = profile/ np.sum(profile)
                else: outProf = profile

                winArray[ww] = outProf
                ww+=1
                
                #p = self.mybatch.findProfileStats(profile)
                p = self.mybatch.findMomentStats(profile)
                centList.append(p[1])
                sigList.append(p[2])

            #Append that whole list as one timepoint
            #slideProfiles.append(np.asarray(winList))
            slideProfiles.append(winArray)
            slideCentroids.append(np.asarray(centList))
            slideSigmas.append(np.asarray(sigList))
            #plt.pcolormesh(winArray)
            #plt.show()


            bar.increment()
            bar.display()
        bar.display(force = True)  

        #Convert to arrays
        self.slideArray = np.asarray(slideProfiles)
        self.slideCents = np.asarray(slideCentroids)
        self.slideSigs = np.asarray(slideSigmas)

        #Get statistics on centroids
        self.centDt = 100
        varCentsL = []
        for tt in self.times:
            range = self.makeRange(tt,self.centDt,NT)

            centroids = self.slideCents[range]
            varience = np.std(centroids, axis = 0)
            varCentsL.append(varience)
        self.varCents = np.asarray(varCentsL)

        self.cents = np.std(self.slideCents, axis = 0)

    def getWindow(self, profiles, tt, window, NT):
        """Get the binned profile at given time with given window"""
        range = self.makeRange(tt, window, NT)
        profilez = np.sum(profiles[range][:], axis = 0)
        return profilez

    def getMaxWindow(self, profiles, tt, window, NT):
        range = self.makeRange(tt, window, NT)
        return profiles[range][:]

    def makeRange(self, tt, window, NT):
        range = np.arange(-window, 1) + tt
        range = np.mod(range, NT).astype(int)
        return range

        #start = tt-window
        #end = tt + 1
        
        ##low = math.fmod(tt-window, NT)
        #low = np.mod(tt-window, -NT)
        #high = np.mod(tt+1, NT)
        ##if low > high: low, high = high, low
        #range = np.arange(low,high)

    def windowPlot(self):
        
        #Create all the axes
        from matplotlib import gridspec
        gs = gridspec.GridSpec(1,3, width_ratios=[3,3,1])
        fig = plt.figure()
        mAx = plt.subplot(gs[0])
        slAx= plt.subplot(gs[1], sharex = mAx)
        vAx = plt.subplot(gs[2], sharey = slAx)

        #fig, [mAx, slAx, vAx] = plt.subplots(1,3, sharey = True, gridspec_kw = {'width_ratios':[1, 4, 1]})
        tit = fig.suptitle("Time: {}".format(self.firstIndex))


        #Just the regular time series with ticker line
        vmax = np.amax(self.lineArray) * 0.7
        vmin = np.amin(self.lineArray) * 0.9
        mAx.pcolormesh(self.env.lamAx.astype('float32'), self.times, self.lineArray, vmin = vmin, vmax = vmax)
        slide = 0.2
        mAx.set_xlim(self.env.lam0 - slide, self.env.lam0 + slide)
        mAx.set_ylabel('Time')
        mAx.set_title('Time Series')
        tickerLine = mAx.axhline(0)

        #The sliding window plot
        slAx.set_title('Integration View')
        slAx.set_xlabel("Wavelength")
        slAx.set_ylabel("Integration Time (S)")
        slAx.set_ylim(0, np.amax(self.winAx))
        quad = slAx.pcolormesh(self.env.lamAx, self.winAx, self.slideArray[self.lastIndex])

        #The centroid line
        line1, = slAx.plot(self.slideCents[self.lastIndex], self.winAx, 'r')
        slAx.axvline(self.env.lam0, color = 'k', ls= ":")

        #THe sigma lines
        line2, = slAx.plot(self.slideCents[self.lastIndex] + self.slideSigs[self.lastIndex], self.winAx, 'm')
        line3, = slAx.plot(self.slideCents[self.lastIndex] - self.slideSigs[self.lastIndex], self.winAx, 'm')

        #Centroid Sigma
        vCents = self.mybatch.std2V(self.cents)
        self.vSigs = self.mybatch.std2V(self.slideSigs)
        vAx.plot(vCents, self.winAx)
        vAx.axvline(color = 'k')
        vAx.set_title("St. Dev. of the Centroid")
        #line4, = vAx.plot(self.vSigs[self.lastIndex], self.winAx)
        #throw = np.nanmax(self.varCents)
        #vAx.set_xlim(0,throw)
        #vAx.set_title('Varience of the Centroid \nfor {}s'.format(self.centDt))
        vAx.set_xlabel('km/s')


        def init():
            tit.set_text("Time: {}".format(self.lastIndex))
            tickerLine.set_ydata(self.lastIndex)
            quad.set_array(self.slideArray[self.lastIndex][:-1,:-1].flatten())
            line1.set_xdata(self.slideCents[self.lastIndex])
            line2.set_xdata(self.slideCents[self.lastIndex] + self.slideSigs[self.lastIndex])
            line3.set_xdata(self.slideCents[self.lastIndex] - self.slideSigs[self.lastIndex])
            #line4.set_xdata(self.vSigs[self.lastIndex])
            return tickerLine, quad, line1, line2, line3, tit

        #Animate
        from matplotlib import animation

        def animate(i):
            tit.set_text("Time: {}".format(i))
            tickerLine.set_ydata(i)
            quad.set_array(self.slideArray[i][:-1,:-1].flatten())
            line1.set_xdata(self.slideCents[i])
            line2.set_xdata(self.slideCents[i] + self.slideSigs[i])
            line3.set_xdata(self.slideCents[i] - self.slideSigs[i])
            #line4.set_xdata(self.vSigs[i])
            return tickerLine, quad, line1, line2, line3

        anim = animation.FuncAnimation(fig, animate, init_func = init, frames = np.arange(self.firstIndex,self.lastIndex), 
                                       repeat = True, interval = 75, blit=True)
        
        grid.maximizePlot()
        plt.show(False)
        plt.close()
        anim.save(filename = self.movName, writer = 'ffmpeg', bitrate = 1000)
        print('Save Complete')
        #plt.tight_layout()
        #plt.show()


        return     

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
        cent_ax.axvline(self.env.lam0)

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


        self.initLists()

        if self.comm.size > 1:
            self.MPI_init_masterslave()
        else: self.init_serial()
        #self.findProfiles()

    def init_Masterlines(self):
        self.masterLine = []
        ind = 0
        grd=grid.defGrid().primeLineLong
        for env in self.envs:
            grd.reset()
            #simulation = simulate(grd, self.envs[ind], self.N, findT = self.findT, timeAx = self.timeAx, printOut = self.printSim, getProf = True)
            self.masterLine.append(simulate(grd, self.envs[ind], self.N, findT = False, timeAx = self.timeAx, printOut = self.printSim, getProf = True)) 
            ind += 1

    def MPI_init_masterslave(self):

        if self.root and self.print: 
            print('Running MultiSim: ' + time.asctime())
            t = time.time() #Print Stuff
            print('Nenv = ' + str(self.Nenv), end = '; ')
            print('Lines\Env = ' + str(len(self.oneBatch)), end = '; ')
            print('JobSize = ' + str(len(self.batch)), end = '; ')

            print('PoolSize = ' + str(self.size), end = '; ')
            #print('ChunkSize = ' + str(len(self.gridList)), end = '; ') 
            #print('Short Cores: ' + str(self.size * len(self.gridList) - len(self.batch)))#Print Stuff
            print('')
            self.Bar = pb.ProgressBar(len(self.batch))
            self.Bar.display()
        else: self.Bar = None

        if self.useMasters: self.init_Masterlines()
        

        work = [[bat,env] for bat,env in zip(self.batch, self.envInd)]

        self.poolMPI(work, self.mpi_sim)

        try: 
            self.Bar.display(force = True)
            sys.stdout.flush()
        except: pass

        if self.destroySims and self.root: self.sims = self.sims[0:1]


    def initLists(self):
        self.sims = []
        self.profilesC = []
        self.intensitiesC = []
        self.profilesR = []
        self.intensitiesR = []
        self.pBs = []
        self.urProjs = []
        self.rmsProjs = []
        self.temProjs = []
        self.indices = []

    def collectVars(self, simulation):
        if self.keepAll: self.sims.append(simulation)
        else: self.sims = [simulation]
        self.profilesC.append(simulation.profilesC)
        self.intensitiesC.append(np.sum(simulation.profilesC))
        self.profilesR.append(simulation.profilesR)
        self.intensitiesR.append(np.sum(simulation.profilesR))
        self.pBs.append(simulation.pB)
        self.urProjs.append(simulation.urProj)
        self.rmsProjs.append(simulation.rmsProj)
        self.temProjs.append(simulation.temProj)
        self.indices.append(simulation.index)

    def mpi_sim(self, data):
        grd, envInd = data

        if self.useMasters:
            simulation = self.workrepoint(grd, envInd)
        else:
            simulation = simulate(grd, self.envs[envInd], self.N, findT = self.findT, timeAx = self.timeAx, printOut = self.printSim, getProf = True)
            #simulation.plot('nion', cmap = 'RdBu', center = True, abscissa = 'pPos')

        return simulation

    def workrepoint(self, grd, envInd):
        return self.masterLine[envInd].repoint(grd)

            
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
        self.init_Masterlines()
        work = [[bat,env] for bat,env in zip(self.batch, self.envInd)]
        self.sims = []
        self.Bar = pb.ProgressBar(len(work))

        for line in work:
            self.collectVars(self.mpi_sim(line))
            self.Bar.increment()
            self.Bar.display()

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
        profilesC = self.profilesC 
        profilesR = self.profilesR

        for field in self.dict():
            delattr(self, field)

        self.profilesC = profilesC
        self.profilesR = profilesR
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
            self.collectVars(data)
            all_data.append([])
            try:
                self.Bar.increment() 
                self.Bar.display()
            except:pass
            comm.send(anext, dest=status.Get_source(), tag=WORKTAG)

 
        for i in range(1,size):
            data = comm.recv(None, source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
            self.collectVars(data)
            all_data.append([])
            try:
                self.Bar.increment() 
                self.Bar.display()
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
            result = do_work(data)
            del result.comm
            comm.send(result, dest=0)
    
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
        print("Loading Batch: {}".format(self.batchName))
        batchPath = '../dat/batches/{}.batch'.format(self.batchName)
        absPth = os.path.normpath(batchPath)
        try:
            with open(absPth, 'rb') as input:
                return pickle.load(input)
        except:
            sys.exit('Batch Not found')

    def restartBatch(self, envs):
        myBatch = self.loadBatch()
        myBatch.reloadEnvs(envs)
        myBatch.findRank()
        myBatch.simulate_now()
        return myBatch

    def analyzeBatch(self):
        myBatch = self.loadBatch()
        myBatch.findRank()
        myBatch.doStats()
        return myBatch


class batchjob:
    qcuts = [16,50,84]
    ## Run the Simulation ######################################################
    def __init__(self, envs):
    
        if type(envs) is list or type(envs) is tuple: 
            self.envs = envs
        else: self.envs = [envs]
        #print(self.envs)
        self.env = self.envs[0]
        self.ions = self.env.getIons(self.env.maxIons)
        self.firstRunEver = True
        self.complete = False
        self.completeTime = "Incomplete Job"
        comm = MPI.COMM_WORLD
        self.root = comm.rank == 0
        try: self.intTime = len(self.timeAx)
        except: self.intTime = np.nan


        #self.rmsPlot()
        #return
        self.simulate_now()
        
        self.finish()

    def simulate_now(self):

        comm = MPI.COMM_WORLD

        if self.root and self.print:
            print('\nCoronaSim!')
            print('Written by Chris Gilbert')
            print('-------------------------\n')
            print("Simulating Impacts: {}".format(self.labels))
            print("Ions: {}".format(["{}_{}".format(x['ionString'], x['ion']) for x in self.ions]))
            print("Integration Time: {} seconds".format(len(self.timeAx)))
            sys.stdout.flush()

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
            self.bar.display(True)
            sys.stdout.flush()

        while len(self.doLabels) > 0:
            ind = self.doLabels.pop(0)
            self.doneLabels.append(ind)
            thisBatch = self.batch.pop(0)
            try:
                self.count += 1
            except:
                self.count = 1
            if self.root and self.printMulti: 
                #print('\n\n\n--' + self.xlabel +' = ' + str(ind) + ': [' + str(self.count) + '/' + str(self.Nb) + ']--') 
                print('\n\n\n--{} = {}: [{}/{}]--'.format(self.xlabel, ind, self.count, self.Nb))

            if self.autoN:
                try: N = self.bRez.pop(0)
                except:
                    N = self.N
                    print('Auto Rez Failed')
                
            else:
                N = self.N
            
            thisSim = multisim(thisBatch, self.envs, N, printOut = self.printMulti, printSim = self.printSim, timeAx = self.timeAx)
            
            comm.barrier()

            if self.root:
                self.collectVars(thisSim)

                if self.print:
                    if self.printMulti: print('\nBatch Progress: '+ str(self.batchName))
                    self.bar.increment()
                    self.bar.display(True)
                if self.firstRunEver: 
                    self.setFirstPoint()
                    self.firstRunEver = False
                self.save(printout = self.print)
                sys.stdout.flush()

            comm.barrier()
            

    def finish(self):
        #del self.env.R_vlos
        if self.root:
            #self.__findBatchStats()
            #self.makeVrms()
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
            self.save(printout = self.print)
 
    def stop(self):
        comm = MPI.COMM_WORLD
        comm.barrier()
        sys.stdout.flush()
        print("slave {} reporting".format(self.rank))
        sys.stdout.flush()
        import pdb
        pdb.set_trace()

    def setFirstPoint(self):
        self.copyPoint = self.sims[0].sims[0].sims[0]

    def initLists(self):
            self.sims = []
            self.profilessC = []
            self.intensitiessC = []
            self.profilessR = []
            self.intensitiessR = []
            self.indicess = []
            
            self.pBss = []
            self.urProjss = []
            self.rmsProjss = []
            self.temProjss = []

    def collectVars(self, thisSim):
        self.sims.append(thisSim)

        self.profilessC.append(thisSim.profilesC)
        self.intensitiessC.append(thisSim.intensitiesC)
        self.profilessR.append(thisSim.profilesR)
        self.intensitiessR.append(thisSim.intensitiesR)
        
        self.indicess.append(thisSim.indices)
        self.pBss.append(thisSim.pBs)
        self.urProjss.append(thisSim.urProjs)
        self.rmsProjss.append(thisSim.rmsProjs)
        self.temProjss.append(thisSim.temProjs)

    def findRank(self):
        """Discover and record this PU's rank"""
        comm = MPI.COMM_WORLD
        self.rank = comm.Get_rank()
        self.root = self.rank == 0


    ##  Statistics  ##############################################################

    def doStats(self):
        """Do the statistics for the whole batch and plot."""
        try: self.statsDone
        except: self.statsDone = False
        if not self.statsDone or self.redoStats:
            self.makePSF()
            self.__findBatchStats()
        
        self.plot()
        #self.doPB(self.pBname)

    def findProfileStats(self, profile, ion):
        """Analyze a single profile and return a list of the statistics"""
        def gauss_function(x, a, x0, sigma, b): return a*np.exp(-(x-x0)**2/(2*sigma**2)) + b
        #import pdb
        #pdb.set_trace()
        #Convolve with PSF and the deconvolve again
        profileCon, profileDecon = self.conDeconProfile(profile, ion)
        #import pdb
        #pdb.set_trace()
        #Use the moment method to get good initial guesses
        p0 = self.findMomentStats(profile, ion)

        try:
            #Fit gaussians to the lines

            poptRaw, pcovRaw = curve_fit(gauss_function, ion['lamAx'], profile, p0 = p0)  
            poptCon, pcovCon = curve_fit(gauss_function, ion['lamAx'], profileCon, p0 = p0) 
            poptDecon, pcovDecon = curve_fit(gauss_function, ion['lamAx'], profileDecon, p0 = p0)

            ampRaw = np.abs(poptRaw[0])
            muRaw = np.abs(poptRaw[1])
            sigmaRaw = np.abs(poptRaw[2])
            bRaw = np.abs(poptRaw[3])
            perrRaw = np.sqrt(np.diag(pcovRaw)) #one standard deviation errors
            
            ampCon = np.abs(poptCon[0])
            muCon = np.abs(poptCon[1])
            sigmaCon = np.abs(poptCon[2])
            bCon = np.abs(poptCon[3])
            perrCon = np.sqrt(np.diag(pcovCon))

            ampDecon = np.abs(poptDecon[0])
            muDecon = np.abs(poptDecon[1])
            sigmaDecon = np.abs(poptDecon[2])
            bDecon = np.abs(poptDecon[3])
            perrDecon = np.sqrt(np.diag(pcovDecon))

        except (RuntimeError, ValueError):
            return [np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN, np.NaN]

        #Subtract the Point Spread Width in quadrature
        sigmaSubtract = np.sqrt(np.abs(sigmaCon**2 - ion['psfSig_e']**2)) #Subtract off the PSF width


        ionStr = '{}_{} : {} -> {}, $\lambda_0$: {}, #{}'.format(ion['ionString'], ion['ion'], ion['upper'], ion['lower'], ion['lam00'], ion['II'])
        #Plot the fits
        if self.plotFits and self.binCheck:# ion['II'] < self.maxFitPlot: 
            plotax = ion['lamAx'] - ion['lam00']
            fig, ax = plt.subplots()
            ax1 = ax
            ax2 = ax1

            fullint = sum(profile)

            ion['II'] += 1
            
            fig.suptitle(ionStr)
            fig.canvas.set_window_title('{}sec'.format(len(self.timeAx)))

            ax1.plot(plotax, profile/fullint, "g-", label = "Raw")
            ax2.plot(plotax, gauss_function(plotax, ampRaw/fullint, muRaw - ion['lam00'], sigmaRaw, bRaw), 'g.:', label = 
                            "Raw Fit: {:.4} km/s".format(self.std2V(sigmaRaw, ion)))

            #plt.plot(self.lamAx, profileCon, "b", label = "Convolved")
            #plt.plot(self.lamAx, gauss_function(self.lamAx, ampCon, muCon, sigmaCon, bCon), 'b.:', label =
            #         "Conv Fit: {:.4} km/s".format(self.std2V(sigmaCon)))

            #plt.plot(self.lamAx, gauss_function(self.lamAx, ampCon, muCon, sigmaSubtract, bCon), 'c.:', label = 
            #         "Subtraction: {:.4} km/s".format(self.std2V(np.abs(sigmaSubtract))))
            ax2.plot(plotax, gauss_function(plotax, ampRaw/fullint, muCon - ion['lam00'], sigmaSubtract, bRaw), 'm.:', label = 
                        "Subtraction {:.4} km/s".format(self.std2V(np.abs(sigmaSubtract), ion)))
 
            #plt.plot(self.lamAx, profileDecon, "r--", label = "Deconvolved")
            #plt.plot(self.lamAx, gauss_function(self.lamAx, ampDecon, muDecon, sigmaDecon, bDecon), 'r.:', label = 
            #         "Deconvolved Fit: {:.4} km/s".format(self.std2V(sigmaDecon)))
            ax2.set_xlim([-2,2])
            ax2.set_ylim([0,0.03])
            #grid.maximizePlot()
            #plt.yscale('log')
            ax2.set_title("A Line at b = {}".format(self.impact))
            ax2.set_xlabel('Wavelength (A)')
            ax2.set_ylabel('Intensity (Arb. Units)')
            ax2.legend()
            plt.show()
        try:
            if self.plotbinFitsNow:
                fig = ion['binfig']
                axarray = ion['binax']
                status = ion['binstatus']
                ax1 = axarray[status]
                ax2 = axarray[status+2]

                fig.suptitle(ionStr)
                ax1.plot(ion['lamAx'], profile/np.sum(profile), "g", label = "Raw")


                fitprofile = gauss_function(ion['lamAx'], ampRaw, muCon, sigmaSubtract, bRaw)
                normprofile = fitprofile/np.sum(fitprofile)
                ax2.plot(ion['lamAx'], normprofile, 'm', label = 
                        "Subtraction {:.4} km/s".format(self.std2V(np.abs(sigmaSubtract), ion)))
        except:pass
        #Select which of the methods gets output
        if self.usePsf:
            if self.reconType.casefold() in 'Deconvolution'.casefold():
                self.reconType = 'Deconvolution'
                ampOut, muOut, sigOut, perrOut = ampDecon, muDecon, sigmaDecon, perrDecon
            elif self.reconType.casefold() in 'Subtraction'.casefold():
                self.reconType = 'Subtraction'
                ampOut, muOut, sigOut, perrOut = ampCon, muCon, sigmaSubtract, perrCon
            elif self.reconType.casefold() in 'None'.casefold():
                self.reconType = 'None'
                ampOut, muOut, sigOut, perrOut = ampCon, muCon, sigmaCon, perrCon
            else: raise Exception('Undefined Reconstruction Type')
        else:
            self.reconType = 'N/A'
            ampOut, muOut, sigOut, perrOut = ampRaw, muRaw, sigmaRaw, perrRaw

        #Check the reconstruction against the raw fit
        ratio = self.std2V(sigOut, ion) / self.std2V(sigmaRaw, ion) 


        return [ampOut, muOut, sigOut, 0, 0, perrOut, ratio]


    def makePSF(self):
        """Do all the things for the PSF that only have to happen once per run"""
        def gauss_function(x, a, x0, sigma, b): return a*np.exp(-(x-x0)**2/(2*sigma**2)) + b
        for ion in self.ions:
            lamRez = np.diff(ion['lamAx'])[0]
            ion['psfSig_e'] = self.FWHM_to_e(ion['psfsig_FW'])
            psfPix = int(np.ceil(ion['psfSig_e']/lamRez))
            ion['psf'] = gauss_function(ion['lamAx'], 1, ion['lam00'], ion['psfSig_e'], 0)
        
            pass

    def FWHM_to_e(self, sig): return sig / 2*np.sqrt(np.log(2))
    def e_to_FWHM(self, sig): return sig * 2*np.sqrt(np.log(2))

    def conDeconProfile(self, profile, ion):
        """Convolve a profile and deconvolve it again, return both. """
        profLen = len(profile)

        buffer = 0.1

        #Pad the profile and the psf, both for resolution and for edge effects
        padd = 1000 #self.psfPix 
        psfPad = np.pad(ion['psf'], padd, 'constant')
        profilePad = np.pad(profile, (0,2*padd), 'constant')

        #Shift the PSF to be correct in frequency space and normalize it
        psfShift = np.fft.fftshift(psfPad)  
        psfShift = psfShift/np.sum(psfShift)

        #Transform the PSF
        psFFT = np.fft.fft(psfShift)

        #Convolve
        proFFT = np.fft.fft(profilePad)
        convFFT =  proFFT * psFFT
        profileCon = np.fft.ifft(convFFT)

        #Deconvolve
        proConFFT = np.fft.fft(profileCon)
        dconvFFT = proConFFT / (psFFT + buffer)
        profileDecon = np.fft.ifft(dconvFFT)

        if self.plotFits and ion['II'] < self.maxFitPlot and False:
            plt.figure()
            plt.plot(np.fft.ifftshift(psFFT), label='psf')
            plt.plot(np.fft.ifftshift(proFFT), ':', label='profile')
            plt.plot(np.fft.ifftshift(convFFT), label='conv')
            plt.plot(np.fft.ifftshift(dconvFFT), label='dconv')
            plt.legend()
            grid.maximizePlot()
            plt.show(False)


        #psf = self.con.Gaussian1DKernel(pix)
        #padleft = (len(profile) - len(self.psf.array)) // 2
        #padright = padleft + (len(profile) - len(self.psf.array)) % 2 
        #psfLong = np.pad(self.psf.array, (padleft, padright), mode='constant')
        ##psfLong = np.fft.fftshift(psfLong)
        #diff =  self.lamRez 
        #padd = int(np.ceil(angSig/diff))

        #profileCon = self.con.convolve(profilePad, np.fft.ifftshift(psfShift[:-1]), boundary='extend', normalize_kernel = True)
        #profileDecon, remainder = scisignal.deconvolve(profileCon, psfShift)
        #padleft = (len(profile) - len(profileDecon)) // 2
        #padright = padleft + (len(profile) - len(profileDecon)) % 2
        #profileDecon = np.pad(profileDecon, (padleft, padright), mode='constant')
        #plt.plot(self.env.lamAx, profile, label="raw")
        #plt.plot(self.env.lamAx, profileCon[0:profLen], label="con")
        #plt.plot(self.env.lamAx, profileDecon[0:profLen], ":", label="decon")
        ##plt.plot(self.env.lamAx, remainder[0:profLen], ":", label="rem")
        #plt.legend()
        #plt.figure()
        #plt.plot(prcFFT)
        #plt.show()

        return np.abs(profileCon[0:profLen]), np.abs(profileDecon[0:profLen])

    def findMomentStats(self, profile, ion):
        #Finds the moment statistics of a profile

        maxMoment = 5
        moment = np.zeros(maxMoment)
        for mm in np.arange(maxMoment):
                moment[mm] = np.dot(profile, ion['lamAx']**mm)

        powerM = moment[0] 
        muM = moment[1] / moment[0]
        sigmaM = np.sqrt(moment[2]/moment[0] - (moment[1]/moment[0])**2)
        skewM = (moment[3]/moment[0] - 3*muM*sigmaM**2 - muM**3) / sigmaM**3
        kurtM = (moment[4] / moment[0] - 4*muM*moment[3]/moment[0]+6*muM**2*sigmaM**2 + 3*muM**4) / sigmaM**4 - 3
        ampM = powerM/(np.sqrt(2*np.pi) * sigmaM)/len(ion['lamAx'])

        return [ampM, muM, sigmaM, 0]

    def __findSampleStats(self):
        #Finds the mean and varience of each of the statistics for each multisim
        nIon = len(self.ions)
        self.stat = []
        self.statV = []
        self.allStd = []
        self.binStdV = []
        self.binStdVsig = []
        self.allRatio = []
        for ii in np.arange(nIon):
            self.stat.append([[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]])
            self.statV.append([[[],[]],[[],[]],[[],[]],[[],[]],[[],[]],[[],[]]])
            self.allStd.append([])
            self.binStdV.append([])
            self.binStdVsig.append([])
            self.allRatio.append([])

        for simStats, impact, binStats in zip(self.impactStats, self.labels, self.binStats):
        #For Each Impact Parameter

            for ion, binIon, idx in zip(self.ions, binStats, np.arange(nIon)):
            #For Each Ion
                ion['idx'] = idx

                #Collect all of the measurements
                allAmp = [x[idx][0] for x in simStats if not np.isnan(x[idx][0])]
                allMean = [x[idx][1] for x in simStats if not np.isnan(x[idx][1])]
                allMeanC = [x[idx][1] - ion['lam00'] for x in simStats if not np.isnan(x[idx][1])]
                allStd = [x[idx][2] for x in simStats if not np.isnan(x[idx][2])]
                allSkew = [x[idx][3] for x in simStats if not np.isnan(x[idx][3])]
                allKurt = [x[idx][4] for x in simStats if not np.isnan(x[idx][4])]
                allRatio = [x[idx][6] for x in simStats if not np.isnan(x[idx][6])]
            
                #Wavelength Units
                self.assignStat(idx, 0, allAmp)
                self.assignStat(idx, 1, allMeanC)
                self.assignStat(idx, 2, allStd)
                self.assignStat(idx, 3, allSkew)
                self.assignStat(idx, 4, allKurt)
            
                #Velocity Units
                self.assignStatV(idx, 0, allAmp)
                self.assignStatV(idx, 3, allSkew)
                self.assignStatV(idx, 4, allKurt)

                #I think that these functions do what they are supposed to.
                mean1= self.__mean2V(np.mean(allMean), ion)
                std1 = np.std([self.__mean2V(x, ion) for x in allMean])
                self.assignStatV2(idx, 1, mean1, std1)

                self.allStd[idx].append([self.std2V(x, ion) for x in allStd])
                mean2= self.std2V(np.mean(allStd), ion)
                std2 = self.std2V(np.std(allStd), ion)
                self.assignStatV2(idx, 2, mean2, std2)

                #Collect the binned version of the profile stats
                try:
                    s1 = binIon[2]
                    s2 = binIon[5][2]
                except:
                    s1, s2 = np.nan, np.nan

                self.binStdV[idx].append(self.std2V(s1, ion)) 
                self.binStdVsig[idx].append(self.std2V(s2, ion)) 

                self.allRatio[idx].append(allRatio)
                self.assignStat(idx, 5, allRatio)
                self.assignStatV(idx, 5, allRatio)

    def assignStat(self, idx, n, var):
        self.stat[idx][n][0].append(np.mean(var))
        self.stat[idx][n][1].append(np.std(var))

    def assignStatV(self, idx, n, var):
        self.statV[idx][n][0].append(np.mean(var))
        self.statV[idx][n][1].append(np.std(var))

    def assignStatV2(self, idx, n, mean, std):
        self.statV[idx][n][0].append(mean)
        self.statV[idx][n][1].append(std)


    def __mean2V(self, mean, ion):
        #Finds the redshift velocity of a wavelength shifted from lam0
        return self.env.cm2km((self.env.ang2cm(mean) - self.env.ang2cm(ion['lam00'])) * self.env.c / 
                (self.env.ang2cm(ion['lam00'])))

    def std2V(self, std, ion):
        return np.sqrt(2) * self.env.cm2km(self.env.c) * (std / ion['lam00'])

    def __findBatchStats(self):
        """Find the statistics of all of the profiles in all of the multisims"""
        self.impactStats = []
        self.binStats = []
        self.intensityStats = []
        print("Fitting Profiles...")
        bar = pb.ProgressBar(len(self.profilessC))
        bar.display()

        
        if self.plotbinFits:
            for ion in self.ions:
                ion['binfig'], binax = plt.subplots(2,2, sharex = True)
                ion['binax'] = binax.flatten()
                ii = 0
                titles = ['Binned Line', 'All Lines', 'Binned Fit', 'All Fits']
                for ax in binax.flatten():
                    ax.set_title(titles[ii])
                    ii+=1

        self.plotbinFitsNow = False
        
        
        for impProfilesC, impProfilesR, impact in zip(self.profilessC, self.profilessR, self.doneLabels):
            #For each impact parameter
            self.impact = impact

            #Plot stuff
            for ion in self.ions:
                ion['II'] = 0 if impact > self.plotheight else 1
            self.plotbinFitsNow = True if impact > self.plotheight and self.plotbinFits else False

            #Find Statistics
            impProfilesS = self.mergeImpProfs(impProfilesC,impProfilesR) #Sum up the collisional and resonant componants
            self.impactStats.append(self.__findSimStats(impProfilesS)) #Find statistics for each line
            self.binStats.append(self.__findStackStats(impProfilesS)) #Find statistics for binned lines
            self.intensityStats.append(self.__findIntensityStats(impProfilesC,impProfilesR)) #Find statistics for binned lines

            ##Plot the line componants and sum
            #plt.plot(impProfilesC[0][0], label = "C")
            #plt.plot(impProfilesR[0][0], label = "R")
            #plt.plot(impProfilesS[0][0], label = "S")
            #plt.legend()
            #plt.show()

            #Plot Stuff
            if impact > self.plotheight and self.plotbinFits:
                for ion in self.ions:
                    ion['binfig'].show()


            bar.increment()
            bar.display()
        bar.display(force = True)
        self.__findSampleStats()
        self.statsDone = True

        if self.plotbinFits:
            pass
            #for ion in self.ions:
            #    plt.close(ion['binfig'])
        else: self.save(printout = True)

    def __findStackStats(self, impProfiles):
        """Stack all the profiles in a list and then get statistics"""

        stacks = []
        for ion in self.ions:
            ion['binstatus'] = 0
            stacks.append(np.zeros_like(ion['lamAx']))

        for profileBundle in impProfiles:
            for ionLine, stack in zip(profileBundle, stacks):
                stack += ionLine #/ np.sum(ionLine)

        self.binCheck = True
        out = self.__findIonStats(stacks)
        self.binCheck = False
        return out

    def __findSimStats(self, impProfiles):
        """Find the statistics of all profiles in a given list"""
        simStats = []
        self.binCheck = False
        for ion in self.ions:
            ion['binstatus'] = 1
        for profileBundle in impProfiles:
            simStats.append(self.__findIonStats(profileBundle))
        return simStats

    def __findIonStats(self, profileBundle):
        """Finds the stats for each ion in a given LOS"""
        ionStats = []
        for ion, ionLine in zip(self.ions, profileBundle):
            ionStats.append(self.findProfileStats(ionLine, ion))
        return ionStats

    def __findIntensityStats(self,impProfilesC,impProfilesR):
        """cant do it"""
        nions = len(self.ions)
        Cbucket = np.zeros(nions)
        Rbucket = np.zeros(nions)
        ii = 0

        returnlist = []
        for bundleC, bundleR in zip(impProfilesC, impProfilesR):
            #For each LOS
            ii+=1
            jj = 0
            for ion, profC, profR in zip(self.ions, bundleC, bundleR):
                dlam = np.mean(np.diff(ion['lamAx']))
                #For each ion
                Cbucket[jj] += np.sum(profC) * dlam
                Rbucket[jj] += np.sum(profR) * dlam
                jj += 1

        avgC = Cbucket / ii
        avgR = Rbucket / ii
        thisResult = (avgC, avgR)

        return thisResult #a list for each ion, the two intensities

    def mergeImpProfs(self, impProfilesC, impProfilesR):
        #Sum two different boxes of profiles
        A = np.asarray(impProfilesC)
        B = np.asarray(impProfilesR)
        C = np.zeros_like(impProfilesC)

        if self.resonant and self.collisional:
            #if both, sum
            for ii in np.arange(len(A)):
                for jj in np.arange(len(A[0])):
                    C[ii][jj]=A[ii][jj]+B[ii][jj]
            return C.tolist()
        elif self.collisional:  return A.tolist()
        elif self.resonant: return B.tolist()
        else:  return C.tolist()
        

    def plotProfiles(self, max):
        if max is not None:
            for profiles, impact in zip(self.profiless, self.doneLabels):
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

    ## Main Plots ########################################################################
    def plot(self):
        if self.pIon: self.ionPlot()
        self.massPlot()
        if self.pProportion: self.plotStack(self.ions[self.plotIon])
        if self.pWidth: self.plotMultiWidth()
        if self.pPB: self.plotPB()
        if self.pIntRat: self.plotIntRat()
        plt.show(block=True)


    def plotIntRat(self):
        """Plot the average intensity and the CvsR proportion"""
        labs = self.doneLabels
        
        #unpack the data
        C,R  = zip(*self.intensityStats)

        Cions = list(zip(*C))
        Rions = list(zip(*R))

        for ion, cint, rint in zip(self.ions, Cions, Rions):
            #For each ion
            tint = [c+r for c,r in zip(cint,rint)]

            cnorm = np.asarray(cint)/np.asarray(tint)
            rnorm = np.asarray(rint)/np.asarray(tint)

            fig, (ax0,ax1) = plt.subplots(2, 1, True)
            plt.figtext(0.1,0.95,self.batchName)
            ax0.set_title('Normalized Contribution')
            ax0.plot(labs, cnorm, 'b', label='Collisional')
            ax0.plot(labs, rnorm, 'r', label='Resonant')

            ax0.axhline(1, c='k')
            ax0.axhline(0.5, c='k', ls=':')
            ax0.axhline(0, c='k')

            ax0.legend()
            fig.suptitle('{}_{}'.format(ion['ionString'], ion['ion']))

            ax1.plot(labs, cint, 'b', label='Collisional')
            ax1.plot(labs, rint, 'r', label='Resonant')
            ax1.plot(labs, tint, 'k', label='Total')
            ax1.set_title('Absolute Total Intensity')
            ax1.set_yscale('log')
            ax1.set_ylabel(r'$ergs/(s\ cm^2\ sr)$')
            ax1.legend()

            ax1.set_xlabel(r'Impact Parameter $b/R_\odot$')
            if self.pIntRat == 'save': plt.savefig('../fig/2018/ratios/{}_{}_{}.png'.format(ion['ionString'], ion['ion'], ion['lam00']))
            else: plt.show()
            plt.close(fig)
            

        #import pdb; pdb.set_trace()

            #cint, rint = zip(*stat)

            #tint = [c+r for c,r in zip(cint,rint)]
            #frac = [c/t for c,t in zip(cint,tint)]
        
            
        
        
        pass

    def plotPB(self):
        def pbFit(r): return 1e-9*(1570*r**-11.1 + 6.75*r**-3.42)
        #sigmaT = 6.65e-25
        #coeff = 3/16*sigmaT


        fig, ax = plt.subplots()

        pbAvg = []
        pbStd = []
        fits = []
        for impact, pbs in zip(self.doneLabels, self.pBss):
            pbAvg.append(np.mean(pbs))
            pbStd.append(np.std(pbs))

            fits.append(pbFit(impact))


        ax.errorbar(self.doneLabels, pbAvg, yerr=pbStd, capsize = 3, label = 'Simulated Values')

        plt.plot(self.doneLabels, [f for f in fits],label= 'Fit Function')

        ax.set_yscale('log')
        ax.set_ylabel('r / $R_\odot$')
        ax.set_ylabel('pB')
        ax.set_title('Polarization Brightness')
        ax.legend()





        plt.show()




        pass

    def plotStatsV(self):
        """Plot all of the statistics for the whole batch."""
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

    def rmsPlot(self):

        #Get the RMS values out of the braid model
        braidRMS = []
        braidImpacts = np.linspace(1.015,3,100)
        for b in braidImpacts:
            braidRMS.append(self.env.extractVrms(b))

        #Get the RMS values out of coronasim
        maxImpact = 20
        rez = 200
        line = simulate(grid.sightline([3,0,0],[maxImpact,0,0], coords = 'Sphere'), self.env, rez, timeAx = [0])
        modelRMS = line.get('vRms')
        modelImpacts = line.get('pPos', 0)

        plt.plot(braidImpacts, [self.env.cm2km(x) for x in braidRMS], label = 'Braid')
        #print(braidRMS[-6])
        #print(braidImpacts[-6])
        plt.plot(modelImpacts, [self.env.cm2km(x) for x in modelRMS], label = 'Model')
        plt.legend()
        plt.xlabel('r / $R_{\odot}$')
        plt.ylabel('RMS Amplitude (km/s)')
        plt.title('RMS Wave Amplitude Extrapolation')
        plt.xscale('log')
        plt.show()

    def getLabels(self):
        try:
            labels = np.asarray(self.doneLabels)
        except: 
            labels = np.arange(len(self.profiles))
            doRms = False
        return labels


    def makeVrms(self, ion):
        self.env._fLoad(self.env.fFileName)
        ion['expectedRms'] = []
        ion['V_wind'] =  []   
        ion['V_waves'] =   []   
        ion['V_thermal'] = [] 
        ion['V_nt'] =[]
        ion['expectedRms_raw'] = []
        ion['V_wind_raw'] =  []   
        ion['V_waves_raw'] =   []   
        ion['V_thermal_raw'] = [] 
        ion['V_nt_raw'] =[]
        self.hahnV = []

        for impact in self.doneLabels:

            ##Get model values for wind, rms, and temperature in plane of sky at this impact
            pTem = self.env.interp_rx_dat(impact, self.env.T_raw) 
            pUr = self.env.interp_rx_dat(impact, self.env.ur_raw) if self.copyPoint.windWasOn else 0
            pRms = self.env.extractVrms(impact) if self.copyPoint.waveWasOn else 0

            vTh = 2 * self.env.KB * pTem / ion['mIon']

            wind =  (self.env.interp_f1(impact) * pUr)**2 
            waves =   (self.env.interp_f2(impact) * pRms)**2
            thermal = (self.env.interp_f3(impact) * vTh)**1

            wind_raw =  (pUr)**2 
            waves_raw =   (pRms)**2
            thermal_raw = (vTh)**1

            V = np.sqrt( (wind + waves + thermal) )
            V_raw = np.sqrt( (wind_raw + waves_raw + thermal_raw) )

            ion['expectedRms'].append(self.env.cm2km(V))
            ion['V_nt'].append(self.env.cm2km(np.sqrt(wind + waves)))
            ion['V_wind'].append(self.env.cm2km(np.sqrt(wind)))
            ion['V_waves'].append(self.env.cm2km(np.sqrt(waves)))
            ion['V_thermal'].append(self.env.cm2km(np.sqrt(thermal)))

            ion['expectedRms_raw'].append(self.env.cm2km(V_raw))
            ion['V_nt_raw'].append(self.env.cm2km(np.sqrt(wind_raw + waves_raw)))
            ion['V_wind_raw'].append(self.env.cm2km(np.sqrt(wind_raw)))
            ion['V_waves_raw'].append(self.env.cm2km(np.sqrt(waves_raw)))
            ion['V_thermal_raw'].append(self.env.cm2km(np.sqrt(thermal_raw)))

            self.hahnV.append(self.hahnFit(impact))

        

    def plotStack(self, ion):
        self.makeVrms(ion)
        fig, ((ax,ax4), (ax2, ax3)) = plt.subplots(2,2, sharex=True)
        fig.suptitle('{} {}: {:.2F}$\AA$'.format(ion['ionString'].capitalize(), self.write_roman(ion['ion']), ion['lam00']))

        xx = self.doneLabels
        
        y1 = [x/y for x,y in zip(ion['V_wind'], ion['expectedRms'])]
        y2 = [x/y for x,y in zip(ion['V_waves'], ion['expectedRms'])]
        y3 = [x/y for x,y in zip(ion['V_thermal'], ion['expectedRms'])]
        ax.set_title('Proportion Compared to Total Line Width')
        ax.set_ylabel('Percentage')
        #ax.set_xlabel('Impact Parameter')
        ax.set_ylim([0,1.05])
        ax.plot(xx, y1, 'b', label = 'Wind')
        ax.plot(xx, y2, 'g', label = 'Waves')
        ax.plot(xx, y3, 'r', label = 'Thermal')
        ax.legend()
        ax.axhline(1, c='k')
        
        ax2.plot(xx, ion['V_wind'], 'b', label='GW Wind')
        ax2.plot(xx, ion['V_waves'], 'g', label='GW Waves')
        ax2.plot(xx, ion['V_thermal'], 'r', label='GW Thermal')
        ax2.plot(xx, ion['expectedRms'], 'k', label='GW Total')
        ax2.plot(xx, ion['V_wind_raw'], 'b:', label='Wind')
        ax2.plot(xx, ion['V_waves_raw'], 'g:', label='Waves')
        ax2.plot(xx, ion['V_thermal_raw'], 'r:', label='Thermal')
        ax2.plot(xx, ion['expectedRms_raw'], 'k:', label='Total')



        ax2.set_xlabel('Impact Parameter')
        ax2.set_ylabel('Weighted Contribution (km/s)', color='b')
        ax2.tick_params('y', colors='b')
        ax2.set_title('Weighted Velocity Components')
        ax2.legend()


        ax3.plot(xx, self.width2T(ion, ion['V_wind']), 'b', label='GW Wind')
        ax3.plot(xx, self.width2T(ion, ion['V_waves']), 'g', label='GW Waves')
        ax3.plot(xx, self.width2T(ion, ion['V_thermal']), 'r', label='GW Thermal')
        ax3.plot(xx, self.width2T(ion, ion['expectedRms']), 'k', label='Gw Total')

        ax3.plot(xx, self.width2T(ion, ion['V_wind_raw']), 'b:', label='Wind')
        ax3.plot(xx, self.width2T(ion, ion['V_waves_raw']), 'g:', label='Waves')
        ax3.plot(xx, self.width2T(ion, ion['V_thermal_raw']), 'r:', label='Thermal')
        ax3.plot(xx, self.width2T(ion, ion['expectedRms_raw']), 'k:', label='Total')

        ax3.tick_params('y', colors='r')
        ax3.set_xlabel('Impact Parameter')
        ax3.set_ylabel('$T_{eff}$', color='r')
        ax3.set_yscale('log')
        ax3.set_title('Effective Temperature Components')
        ax3.legend()
        
        ax4.set_title('Weighting Functions')
        ax4.plot(xx, [self.env.interp_f1(x) for x in xx], 'b', label='Wind')
        ax4.plot(xx, [self.env.interp_f2(x) for x in xx], 'g', label='Waves')
        ax4.plot(xx, [self.env.interp_f3(x) for x in xx], 'r', label='Thermal')
        ax4.axhline(1, c='k')
        ax4.legend()



        grid.maximizePlot()

        plt.show(False)



    def width2T(self, ion, widths):
        const = ion['mIon']/(2*self.env.KB)*10**10
        temps = [w**2*const for w in widths]
        return temps

    def ionPlot(self):
        fig, ax1 = plt.subplots()
        for ion in self.ions:
            label = '{}_{}'.format(ion['ionString'], ion['ion'])
            ax1.errorbar(self.doneLabels, self.binStdV[ion['idx']], yerr = self.binStdVsig[ion['idx']], fmt = 'o', label = label, capsize=6)
        ax1.legend()
        ax1.set_title('Binned Profile Measurements')
        ax1.set_xlabel('Impact Parameter')
        ax1.set_ylabel('$v_{1/e}$ (km/s)')
        plt.show(False)

    def massPlot(self):
        """Plot each of the ions width vs mass"""
        plot1 = self.pMFit   # V vs Mass
        plot2 = self.pMass    #Temperature and V vs Impact
        twinplot2 = True
        plotFlag = True
        if plot1 or plot2:
            slopes = []
            sloperrors = []
            intercepts = []
            interrors = []

            kb = 1.38064852e-23 #joules/kelvin = kg(m/s)^2/T
            kbg = kb * 1000 #g(m/s)^2/T
            kbgk = kbg / 1e6 #g(km/s)^2/T
            
            if plot1: fig, ax = plt.subplots()
            first = True
            for impact in reversed(np.arange(len(self.doneLabels))):
                widthList = []
                widerList = []
                widthsqList = []
                widthsqerList = []
                invMassList = []
                for ion in self.ions:
                    #Slice the data in the way we want
                    #Retrieve Values
                    wid = self.binStdV[ion['idx']][impact] #(km/s)
                    wider = self.binStdVsig[ion['idx']][impact] #(km/s)
                    invMass = 2*kbgk/ion['mIon'] #(km/s)^2 / T 

                    #Square with error propagation
                    widthsq = wid**2 #(km/s)^2
                    widthsqer = 2*wid*wider #(km/s)^2
                
                    #Store Values
                    widthList.append(wid)
                    widerList.append(wider)
                    widthsqList.append(widthsq) #(km/s)^2
                    widthsqerList.append(widthsqer) #(km/s)^2
                    invMassList.append(invMass) #(km/s)^2 / T 
                if first: locs = widthList
                first = False
                #Fit a line
                weights = [1/w for w in widthsqerList]
                (slope, intercept), cov = np.polyfit(invMassList, widthsqList, 1, 
                                    w = weights, cov = True) #T, (km/s)^2
                (sloper, inter) = np.diag(cov)**0.5

                #Store Values
                slopes.append(slope) #T
                intercepts.append(intercept**0.5) #km/s
                sloperrors.append(sloper)
                interrors.append(inter**0.5)

            #Plotting
                if plot1 and plotFlag: 
                    color = next(ax._get_lines.prop_cycler)['color']
                    #Plot simulated velocities 
                    ax.errorbar(invMassList, widthList, fmt = 'o:', yerr = widerList, label = '{:0.2f}'.format(self.doneLabels[impact]), color = color, capsize = 3)
                    #Plot the fit line
                    ax.plot(invMassList, np.polyval((slope,intercept), invMassList)**0.5, '-', color = color) 
                    #Plot the intercept
                    #ax.plot(0, intercept**0.5, '^', color = color)
                    plotFlag = False
                else: plotFlag = True

        if plot1: 
            #ax.legend(bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure)
            ax.legend(loc=4)
            flag = True
            #for xy in zip(np.zeros_like(self.doneLabels), intercepts): 
            #    #Plot the y intercepts
            #    if flag:
            #        ax.annotate('(%.2f)' % float(xy[1]), xy=xy, textcoords='data')
            #        flag = False
            #    else: flag = True


            lastIon = 'xx'
            for xy, ion in zip(zip([m -2e-5 for m in invMassList], [l+3 for l in locs]), self.ions): 
                #Plot the ion names
                newIon = ion['ionString'][0:2]
                if not newIon == lastIon: ax.annotate('{}'.format(ion['ionString']), xy=xy, textcoords='data')
                lastIon = newIon

            ax.set_xlabel('Inverse Mass (Most Massive to the Left)')
            ax.set_ylabel('Line Width (km/s)')
            ax.set_title('Line Width vs Mass at Each Impact Parameter')
            plt.show(False)

        
        if plot2:
            #Initialize the Plot
            if not twinplot2:
                fig, ax1 = plt.subplots()
                fig.suptitle('Fit parameters as a function of height')
                self.plotLabel()
                fig, ax2 = plt.subplots()
                fig.suptitle('Fit parameters as a function of height')
                self.plotLabel()

            else: 
                fig, (ax1, ax2) = plt.subplots(1,2)
                fig.suptitle('Fit parameters as a function of height')
                self.plotLabel()
            labels = self.doneLabels[::-1]

            #Plot the slopes - the temperatures
            lns1, _, _ = ax1.errorbar(labels, slopes, fmt = 'bo-', yerr = sloperrors, label = 'Fit Temperature', capsize = 4)

            expT = [self.env.interp_rx_dat_log(rx, self.env.T_raw) for rx in labels]
            #gwExpT = [self.env.interp_f3(rx)*self.env.interp_rx_dat_log(rx, self.env.T_raw) for rx in labels]
            lns3 = ax1.plot(labels, expT, 'b--', label = 'POS Model Temp')
            #lns3 = ax1.plot(labels, gwExpT, 'b:', label = 'POS Model Temp')

            #Plot the Effective Temperature
            fig, ax4 = plt.subplots()
            for ion in self.ions:
                #if ion['element'] == 8:
                
                widths = self.binStdV[ion['idx']][::-1]
                yerr = self.binStdVsig[ion['idx']][::-1]
                const = ion['mIon']/(2*self.env.KB)*10**10
                   
                temps = [w**2*const for w in widths]
                xerr = [dx * 2 / x * Q / const for  dx, x, Q in zip(yerr,widths,temps)]
                lns4 = ax4.errorbar(labels, temps, yerr = xerr, fmt = 'o-', label = '{}_{}'.format(ion['ionString'], ion['ion']), capsize=2)
            ax4.set_yscale('log')
            ax4.set_ylim([10**6,10**9])
            ax4.set_xlabel('Impact Parameter')
            ax4.set_ylabel('$T_{eff}$ (Kelvin)')
            ax4.legend(loc = 4)

            ax1.set_yscale('log')
            ax1.set_ylim([10**3,10**8])
            ax1.set_xlabel('Impact Parameter')
            ax1.set_ylabel('Kelvin')

            expWind = [self.env.cm2km(self.env.interp_rx_dat_log(rx, self.env.ur_raw)) for rx in labels]
            lns3 = ax2.plot(labels, expWind, 'c--', label = 'POS Model Wind Speed')

            expWaves = [self.env.cm2km(self.env.findvRms(rx)) for rx in labels]
            lns3 = ax2.plot(labels, expWaves, '--', label = 'POS Model Wave RMS', color = 'orange')

            #expTot = [np.sqrt(x**2 + y**2) for x, y in zip(expWind, expWaves)]
            #lns3 = ax2.plot(labels, expTot, '--', label = 'POS Model Total RMS', color = 'black')

            oneLab1 = "Full"
            oneLab2 = "Thermal"
            oneLab3 = "GW Wind"
            oneLab4 = "GW Waves"
            oneLab5 = "GW Total Non-Thermal"
            oneLab6 = "Thermal"
            #Plot the intercepts - the VRMS
            for ion in self.ions:
                self.makeVrms(ion)
                #lns4 = ax2.plot(self.doneLabels, ion['expectedRms'], 'm:', label = oneLab1)
                #lns4 = ax2.plot(self.doneLabels, ion['V_thermal'], 'c:', label = oneLab2)
                #_    = ax1.plot(self.doneLabels, ion['V_thermal'], 'b:', label = oneLab6)
                #lns4 = ax2.plot(self.doneLabels, ion['V_wind'], 'g:', label = oneLab3)
                #lns4 = ax2.plot(self.doneLabels, ion['V_waves'], 'y:', label = oneLab4)
                lns4 = ax2.plot(self.doneLabels, ion['V_nt'], 'k:', label = oneLab5)
                oneLab1 = None
                oneLab2 = None
                oneLab3 = None
                oneLab4 = None
                oneLab5 = None
                oneLab6 = None

            lns2, _, _ = ax2.errorbar(labels, intercepts, fmt ='ro-', yerr = interrors, label = 'Fit Non-Thermal Velocity', capsize = 4)
            ax2.set_xlabel('Impact Parameter')
            ax2.set_ylabel('km/s')
              
            ax1.legend()
            ax2.legend()
            plt.show(False)

    def plotMultiWidth(self):

        #Determine Shape of Subplot Array
        nIon = len(self.ions)
        sq = np.sqrt(nIon)
        intSq = int(np.floor(sq))
        W = intSq
        H = intSq
        if sq > intSq: H += 1
        if W*H < nIon: W += 1

        #Create Figure
        if False:
            f, axArray = plt.subplots(H, W, sharex = True, sharey = True)
            f.canvas.set_window_title(self.batchName)
        else: 
            axArray = np.asarray([plt.subplots()[1] for ion in self.ions])
            f = axArray[0].figure
        #Format the header
        try: self.completeTime
        except: self.completeTime = 'Incomplete Job'
        
        str1 = "{}: {}\nEnvs: {}; Lines per Env: {}; Lines per Impact: {}\n".format(self.batchName, self.completeTime, self.Nenv, self.Nrot, self.Npt)
        str2 = "usePsf: {}                                          reconType: {}".format(self.usePsf, self.reconType)
        f.suptitle(str1 + str2)

        self.plotLabel()

        #Fill each subplot
        for ion, ax in zip(self.ions, axArray.flatten()):
            self.plotWidth(ion, ax)

        grid.maximizePlot()
        
        plt.show(False)
            
    def plotLabel(self):
        #Display the run flags
        height = 0.9
        left = 0.09
        shift = 0.1
        try: self.intTime
        except: self.intTime = np.nan
        plt.figtext(left, height + 0.08, "Lines/Impact: {}".format(self.Npt))
        plt.figtext(left, height + 0.06, "Seconds: {}".format(self.intTime))
        plt.figtext(left, height + 0.04, "Wind: {}".format(self.copyPoint.windWasOn))
        plt.figtext(left, height + 0.02, "Waves: {}".format(self.copyPoint.waveWasOn))
        plt.figtext(left, height, "B: {}".format(self.copyPoint.bWasOn))    
        plt.figtext(left+ 0.1, height, "Batch: {}".format(self.batchName))     

    def plotWidth(self, ion, ax):
        """Generate the primary plot"""

        ionStr = '{}_{} : {} -> {}, $\lambda_0$: {} $\AA$'.format(ion['ionString'], ion['ion'], ion['upper'], ion['lower'], ion['lam00'])
        ax.set_title(ionStr)

        labels = self.getLabels()

        #Plot the actual distribution of line widths in the background
        histLabels = []
        edges = []
        hists = []
        low = []
        self.medians = []
        high = []

        small = 1e16
        big = 0
        for stdlist in self.allStd[ion['idx']]:
            newBig = np.abs(np.ceil(np.amax(stdlist)))
            newSmall = np.abs(np.floor(np.amin(stdlist)))
            
            small = int(min(small, newSmall))
            big = int(max(big, newBig))
        throw = int(np.abs(np.ceil(big-small))) / 5
        throw = np.arange(0,400,5)
        spread = 2
        vmax = 0
        for stdlist, label in zip(self.allStd[ion['idx']], self.doneLabels):
            
            hist, edge = np.histogram(stdlist, throw)#, range = [small-spread,big+spread])
            hist = hist / np.sum(hist)
            vmax = max(vmax, np.amax(hist))
            histLabels.append(label)
            edges.append(edge)
            hists.append(hist)

            
            quarts = np.percentile(stdlist, self.qcuts)
            low.append(quarts[0])
            self.medians.append(quarts[1])
            high.append(quarts[2])


        array = np.asarray(hists).T
        diff = np.average(np.diff(np.asarray(histLabels)))
        histLabels.append(histLabels[-1] + diff)
        ed = edges[0][:-1]
        xx, yy = np.meshgrid(histLabels-diff/2, ed)
        histLabels.pop()

        hhist = ax.pcolormesh(xx, yy, array, cmap = 'YlOrRd', label = "Sim Hist")
        hhist.cmap.set_under("#FFFFFF")
        hhist.set_clim(1e-8,vmax)
        #cbar = plt.colorbar()
        #cbar.set_label('Number of Lines')

        #Plot the confidence intervals
        ax.plot(histLabels, low, 'c:', label = "{}%".format(self.qcuts[0]), drawstyle = "steps-mid")
        ax.plot(histLabels, self.medians, 'c--', label = "{}%".format(self.qcuts[1]), drawstyle = "steps-mid")
        ax.plot(histLabels, high, 'c:', label = "{}%".format(self.qcuts[2]), drawstyle = "steps-mid")

        #Plot the Statistics from the Lines
        self.lineWidths = self.statV[ion['idx']][2][0]
        self.lineWidthErrors = self.statV[ion['idx']][2][1]
        ax.errorbar(histLabels, self.lineWidths, yerr = self.lineWidthErrors, fmt = 'bo', label = 'Simulation', capsize=4)

        #Do the chi-squared test
        self.makeVrms(ion)  
        self.chiTest(ion)
        #height = 0.9
        #left = 0.65 + 0.09
        #shift = 0.1
        #plt.figtext(left + shift, height + 0.04, "Fit to the Mean")
        #plt.figtext(left + shift, height + 0.02, "Chi2 = {:0.3f}".format(self.chi_mean))
        #plt.figtext(left + shift, height, "Chi2_R = {:0.3f}".format(self.rChi_mean))

        if self.hahnPlot:
            #Plot the Hahn Measurements
            ax.errorbar(self.env.hahnAbs, self.env.hahnPoints, yerr = self.env.hahnError, fmt = 'gs', label = 'Hahn Observations', capsize=4)
            ax.plot(self.doneLabels, self.hahnV, label = "HahnV", color = 'g') 

                         
        #Plot the expected values

        ax.plot(self.doneLabels, ion['expectedRms'], label = 'Expected', color = 'b') 

        #Plot the results of the binned Test
        ax.errorbar(self.doneLabels, self.binStdV[ion['idx']], yerr = self.binStdVsig[ion['idx']], fmt = 'mo', label = "Binned Profiles", capsize=6)





        #Plot Resolution Limit
        diff = ion['lamAx'][1] - ion['lamAx'][0]
        minrez = self.std2V(diff, ion)
        psfrez = self.std2V(ion['psfSig_e'], ion)


        #flr = np.ones_like(self.doneLabels)*minrez
        ax.axhline(minrez, color = 'k', linewidth = 2)
        ax.axhline(psfrez, color = 'k', linewidth = 2, linestyle = ':')
        #plt.plot(self.doneLabels, flr, label = "Rez Limit", color = 'k', linewidth = 2)

        ##Put numbers on plot of widths
        #for xy in zip(histLabels, self.statV[2][0]): 
        #    plt.annotate('(%.2f)' % float(xy[1]), xy=xy, textcoords='data')

        #Put numbers on plot of widths
        #for xy in zip(self.doneLabels, self.binStdV[ion['idx']]): 
        #    ax.annotate('(%.2f)' % float(xy[1]), xy=xy, textcoords='data')


        if ion['idx'] == 0: ax.legend(loc =2)
        if ion['idx'] == len(self.ions)-1: ax.set_xlabel(self.xlabel)
        plt.setp(ax.get_xticklabels(), visible=True)
        ax.set_ylabel('Km/s')
        spread = 0.02
        ax.set_xlim([histLabels[0]-spread, histLabels[-1]+spread]) #Get away from the edges
        ax.set_ylim([0,400])
        
        self.histlabels = histLabels
        if self.plotRatio: self.ratioPlot()

    def ratioPlot(self):
        #Plots the ratio between the raw fits and the reconstructed fits.
        plt.figure()
        histlabels = []
        edges = []
        hists = []

        low = []
        medians = []
        high = []

        small = 1e16
        big = 0
        for ratioList in self.allRatio:
            newBig = np.abs(np.ceil(np.amax(ratioList)))
            newSmall = np.abs(np.floor(np.amin(ratioList)))
            
            small = int(min(small, newSmall))
            big = int(max(big, newBig))
        throw = int(np.abs(np.ceil(big-small)))

        for ratioList, label in zip(self.allRatio, self.doneLabels):
            hist, edge = np.histogram(ratioList, 200, range = [small,big])
            #hist = hist / np.amax(hist)
            histlabels.append(label)
            edges.append(edge)
            hists.append(hist)

            quarts = np.percentile(ratioList, self.qcuts)
            low.append(quarts[0])
            medians.append(quarts[1])
            high.append(quarts[2])

        array = np.asarray(hists).T
        diff = np.average(np.diff(np.asarray(histlabels)))
        histlabels.append(histlabels[-1] + diff)
        ed = edges[0][:-1]
        xx, yy = np.meshgrid(histlabels-diff/2, ed)
        histlabels.pop()

        hhist = plt.pcolormesh(xx, yy, array, cmap = 'YlOrRd', label = "Ratio Hist")
        hhist.cmap.set_under("#FFFFFF")
        hhist.set_clim(1e-8,np.amax(array))
        cbar = plt.colorbar()
        cbar.set_label('Number of Occurances')

        #Plot the Line Ratios
        plt.errorbar(histlabels, self.statV[5][0], yerr = self.statV[5][1], fmt = 'co', label = 'Mean/Std', capsize = 4)

        plt.axhline(1, color = 'k')
        plt.plot(self.histlabels, low, 'c:', label = "{}%".format(self.qcuts[0]), drawstyle = "steps-mid")
        plt.plot(self.histlabels, medians, 'c--', label = "{}%".format(self.qcuts[1]), drawstyle = "steps-mid")
        plt.plot(self.histlabels, high, 'c:', label = "{}%".format(self.qcuts[2]), drawstyle = "steps-mid")

        #Display the run flags
        height = 0.9
        left = 0.09
        shift = 0.1

        plt.figtext(left, height + 0.04, "Wind: {}".format(self.copyPoint.windWasOn))
        plt.figtext(left, height + 0.02, "Waves: {}".format(self.copyPoint.waveWasOn))
        plt.figtext(left, height, "B: {}".format(self.copyPoint.bWasOn))

        str1 = "{}: {}\nEnvs: {}; Lines per Env: {}; Lines per Impact: {}\n".format(self.batchName, self.completeTime, self.Nenv, self.Nrot, self.Npt)
        str2 = "usePsf: {}                                          reconType: {}".format(self.usePsf, self.reconType)
        plt.suptitle(str1 + str2)

        plt.title("Sigma_Reconstructed / Sigma_Raw_Fit")
        plt.ylabel('Ratio')
        plt.xlabel('Impact Parameter')
        plt.legend()
        plt.show()

    def chiTest(self, ion):
                
        #self.hahnMids = np.interp(self.env.hahnAbs, self.histlabels, self.medians)
        #self.hahnMeans = np.interp(self.env.hahnAbs, self.histlabels, self.lineWidths)
        #self.hahnMidErrors = np.interp(self.env.hahnAbs, self.histlabels, self.lineWidthErrors)

        self.chi_bin = 0
        self.chi_mean = 0
        
        N = 0
        for expectedWidth, binWidth, binError, meanWidth, meanError in zip(ion['expectedRms'], self.binStdV[ion['idx']], self.binStdVsig[ion['idx']], self.lineWidths, self.lineWidthErrors):
            N += 1
            self.chi_bin += (binWidth - expectedWidth)**2 / (binError * binError)
            self.chi_mean += (meanWidth - expectedWidth)**2 / (meanError * meanError)
        
        self.rChi_bin = self.chi_bin / N
        self.rChi_mean = self.chi_mean / N

    def getLabels(self):
        try:
            labels = np.asarray(self.doneLabels)
        except: 
            labels = np.arange(len(self.profiles))
            doRms = False
        return labels

    def save(self, batchName = None, keep = False, printout = False, dumpEnvs = True):

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

    def reloadEnvs(self, envs):
        self.envs = envs
            
    def show(self):
        """Print all properties and values except statistics"""
        myVars = vars(self)
        print("\nBatch Properties\n")
        for ii in sorted(myVars.keys()):
            if not "stat" in ii.lower():
                print(ii, " : ", myVars[ii])
                
    def showAll(self):
        """Print all properties and values"""
        myVars = vars(self)
        print("\nBatch Properties\n")
        for ii in sorted(myVars.keys()):
            print(ii, " : ", myVars[ii])

    def doOnePB(self):
        pBs = np.asarray(self.pBss[-1])
        self.pBavg = np.average(pBs)
        self.pBstd = np.std(pBs)


    

    def write_roman(self, num):

        roman = OrderedDict()
        roman[1000] = "M"
        roman[900] = "CM"
        roman[500] = "D"
        roman[400] = "CD"
        roman[100] = "C"
        roman[90] = "XC"
        roman[50] = "L"
        roman[40] = "XL"
        roman[10] = "X"
        roman[9] = "IX"
        roman[5] = "V"
        roman[4] = "IV"
        roman[1] = "I"

        def roman_num(num):
            for r in roman.keys():
                x, y = divmod(num, r)
                yield roman[r] * x
                num -= (r * x)
                if num > 0:
                    roman_num(num)
                else:
                    break

        return "".join([a for a in roman_num(num)])

    def calcFfiles(self):
        #Calculate the f parameters 
        #Make sure to have the B field off and waves and wind on if you want this to work
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

                        #Get the Average Projected Values
                        urProj = np.average(urProjs)
                        rmsProj = np.average(rmsProjs)
                        temProj = np.average(temProjs)

                        #Get the Plane of the Sky Values
                        pTem = self.env.interp_rx_dat(b, self.env.T_raw) 
                        pUr = self.env.interp_rx_dat(b, self.env.ur_raw)
                        pRms = self.env.extractVrms(b)

                        #Take the Quotient
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

                    ##print(rmsProj)
                    ##sys.stdout.flush()
                    ##Simulate a point directly over the pole and average
                    #ptUr = []
                    #ptRms = []
                    #ptTem = []
                    #for env in self.envs:
                    #    point = simpoint([0,0,b], grid = grid.defGrid().impLine, env = env)
                    #    ptUr.append(point.ur)
                    #    ptRms.append(point.vRms)
                    #    ptTem.append(point.T)
                    ##print(ptRms)
                    ##sys.stdout.flush()
                    #pUr = np.average(ptUr)
                    #pRms = np.average(ptRms)
                    #pTem = np.average(ptTem)
                    ##print(pRms)
                    ##sys.stdout.flush()
                    pass

# For doing a multisim at many impact parameters
class impactsim(batchjob):
    def __init__(self, batchName, envs, Nb = 10, iter = 1, b0 = 1.05, b1= 1.50, N = (1500, 10000), 
            rez = None, size = None, timeAx = [0], length = 10, printSim = False, printOut = True, printMulti = True, fName = None, qcuts = [16,50,84], spacing = 'lin'):
        comm = MPI.COMM_WORLD
        self.size = comm.Get_size()
        self.rank = comm.Get_rank() 
        self.root = self.rank == 0
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

        self.print = printOut
        self.printMulti = printMulti
        self.printSim = printSim

        self.N = N

        base = 200

        #if b1 is not None: 
        if b1 is not None: 
            if spacing.casefold() in 'log'.casefold():
                steps = np.linspace(b0,b1,Nb)
                logsteps = base**steps
                logsteps = logsteps - np.amin(logsteps)
                logsteps = logsteps / np.amax(logsteps) * (b1-b0) + b0
                self.labels = logsteps

                #self.labels = np.round(np.logspace(np.log(b0)/np.log(base),np.log(b1)/np.log(base), Nb, base = base), 4)
            else: self.labels = np.round(np.linspace(b0,b1,Nb), 4)
        else: self.labels = np.round([b0], 4)
        #ex = 4
        #xx = np.linspace(0,1,100)
        #kk = xx**ex

        #kmin = 1.0001
        #kmax = 6
        #dif = kmax - kmin
        #qq = kk * dif + kmin
        #self.labels = qq


        self.impacts = self.labels
        self.xlabel = 'Impact Parameter'    
        self.fullBatch = []
        self.bRez = []
        for b in self.impacts:
            long = max(length, 12*b)
            self.bRez.append(long*15)
            self.fullBatch.append(grid.rotLines(N = self.Nrot, b = b, rez = rez, size = size, x0 = long, findT = False)) 


        super().__init__(envs)

        return  
        
        
        


    def hahnFit(self, r, r0 = 1.05, vth = 25.8, vnt = 32.2, H = 0.0657):
        veff = np.sqrt(vth**2+vnt**2 * np.exp(-(r-r0)/(r*r0*H))**(-1/2))
        return veff

class timesim(batchjob):
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

        self.print = printOut
        self.printMulti = printMulti
        self.printSim = printSim

        self.N = N

        #base = 200

        ##if b1 is not None: 
        #if b1 is not None: 
        #    if spacing.casefold() in 'log'.casefold():
        #        steps = np.linspace(b0,b1,Nb)
        #        logsteps = base**steps
        #        logsteps = logsteps - np.amin(logsteps)
        #        logsteps = logsteps / np.amax(logsteps) * (b1-b0) + b0
        #        self.labels = logsteps

        #        #self.labels = np.round(np.logspace(np.log(b0)/np.log(base),np.log(b1)/np.log(base), Nb, base = base), 4)
        #    else: self.labels = np.round(np.linspace(b0,b1,Nb), 4)
        #else: self.labels = np.round([b0], 4)
        
        self.impacts = self.labels
        self.xlabel = 'Impact Parameter'    
        self.fullBatch = []
        for ind in self.impacts:
            self.fullBatch.append(grid.rotLines(N = self.Nrot, b = ind, rez = rez, size = size, x0 = length, findT = False)) 

        super().__init__(envs)
        
        return

class imagesim(batchjob):
    def __init__(self, batchName, env, NN = [5,5], rez=[0.5,0.5], target = [0,1.5], len = 10):
        """Set variables and call the simulator"""
        comm = MPI.COMM_WORLD
        self.size = comm.Get_size()
        self.root = comm.Get_rank() == 0
        self.plotbinFitsNow = False
        try: self.env = env[0]
        except: self.env = env
        #print(self.env)
        self.print = True
        self.printMulti = True
        self.printSim = False
        self.batchName = batchName
        self.labels = np.round([1], 4)
        self.Nb = 1
        #self.N = (200,600)
        self.NN = NN
        self.timeAx = self.timeAx3D #[0]
        self.fName = None

        self.impacts = self.labels
        self.xlabel = 'Nothing'
        self.fullBatch, (self.yax, self.zax) =  grid.image(N = NN, rez = rez, target = target, len = len)
        multisim.destroySims = True


        super().__init__(self.env)

        if self.root: self.imageStats()

        return

    def imageStats(self):
        """Do all the post processing on the lines"""
        print("Finding Line Stats...")
        bar = pb.ProgressBar(len(self.profiless[0]))
        self.makePSF()  

        centroids = []
        sigmas = []
        for profile in self.profiless[0]:
            #Find the centroids and the sigmas
            for ion, prof in zip(self.ions, profile):
                #import pdb
                #pdb.set_trace()
                Stats = self.findProfileStats(prof, ion)
                centroids.append(Stats[1] - ion['lam00'])
                sigmas.append(Stats[2])
            bar.increment()
            bar.display()
        bar.display(True)

        self.centroidss = [centroids]
        self.sigmass = [sigmas]

        #Store all the variables and save
        self.reconstructAll()
        self.save(printout = True)

    def reconstructAll(self):
        """Get out the arrays after they were jumbled from the compute process"""
        self.intensity = self.reconstruct(self.intensitiess)
        self.pB = self.reconstruct(self.pBss)
        self.centroid = self.reconstruct(self.centroidss)
        self.sigma = self.reconstruct(self.sigmass)

    def reconstruct(self, array):
        """Re-order an array to account for random parallel order"""
        indices = [x[2] for x in self.indicess[0]]
        data = array[0]
        out = np.zeros_like(data)
        for ind, dat in zip(indices, data):
            out[ind] = dat
        return out.reshape(self.NN)

    def plot(self):
        #self.indices = [np.array([x[2] for x in self.indicess[0]])]
        #plt.imshow(self.reconstruct(self.indices))
        #plt.show()


        #profiles = self.profiless[0]
        #i = 0
        #for profile in profiles:
        #    plt.plot(profile)
        #    i += 1
        #    #if i > 1000: continue
        #plt.show()

        #int = self.intensitiess[0]
        #hist, edge = np.histogram(int, bins = 100, range = (np.nanmin(int), np.nanmax(int)) )
        #plt.plot(edge[:-1], hist)
        #plt.show()

        #plt.imshow(np.log10(np.array(self.intensitiess[0]).reshape(self.NN)))
        #plt.show()
        #self.reconstructAll()
        #self.save()
        print(self.NN)

        invert = True

        plotInt = True      
        plotpB = False    
        plotStats = False   


        ystring = 'Solar Y ($R_\odot$)'
        zstring = 'Solar Z ($R_\odot$)'
        #self.yax = self.axis[0]
        #self.zax = self.axis[1]
        try: self.centroid
        except: self.imageStats()

        #Process the variables
        centroid = self.centroid
        sigma = self.sigma

        intensityRaw = self.intensity
        intensityCor = self.coronagraph(intensityRaw)
        intensityLog = np.log10(intensityRaw)

        pBRaw = self.pB
        pBCor = self.coronagraph(pBRaw)
        pBLog = np.log10(pBRaw)

        if invert:
            #Invert if Desired
            self.zax, self.yax = self.yax, self.zax
            ystring, zstring = zstring, ystring
            pBRaw, pBCor, pBLog = pBRaw.T, pBCor.T, pBLog.T
            intensityRaw, intensityCor, intensityLog = intensityRaw.T, intensityCor.T, intensityLog.T
            centroid = centroid.T
            sigma = sigma.T

        #Plot

        if plotInt:
            fig0, (ax0, ax1) = plt.subplots(1,2,True, True)
            fig0.set_size_inches(12,9)
            fig0.suptitle(self.batchName)
            intensityLog = np.ma.masked_invalid(intensityLog)
            intensityCor = np.ma.masked_invalid(intensityCor)
            #pdb.set_trace()
            p0 = ax0.pcolormesh(self.zax, self.yax, intensityCor, vmin = 0, vmax = 1)
            ax0.patch.set(hatch='x', edgecolor='black')
            ax0.set_title('Total Intensity - Coronagraph')#\nMIN:{!s:.2} MAX:{!s:.2}'.format(np.nanmin(intensityCor),np.nanmax(intensityCor)))
            ax0.set_ylabel(ystring)
            ax0.set_xlabel(zstring)
            plt.colorbar(p0, ax = ax0)

            p1 = ax1.pcolormesh(self.zax, self.yax, intensityLog, vmin = np.nanmin(intensityLog), vmax = np.nanmax(intensityLog)*0.9)
            ax1.set_title('Total Intensity - Log')
            ax1.patch.set(hatch='x', edgecolor='black')
            ax1.set_xlabel(zstring)
            plt.colorbar(p1, ax = ax1)
            plt.tight_layout()

        if plotpB:
            fig2, (ax4, ax5) = plt.subplots(1,2,True, True)
            fig2.suptitle(self.batchName)
            pBLog = np.ma.masked_invalid(pBLog)
            pBCor = np.ma.masked_invalid(pBCor)

            p0 = ax4.pcolormesh(self.zax, self.yax, pBCor, vmin = 0, vmax = 1)
            ax4.patch.set(hatch='x', edgecolor='black')
            ax4.set_title('Polarization Brightness - Coronagraph\nMIN:{!s:.2} MAX:{!s:.2}'.format(np.nanmin(pBCor),np.nanmax(pBCor)))

            ax5.pcolormesh(self.zax, self.yax, pBLog, vmin = np.nanmin(pBLog), vmax = 6)
            ax5.set_title('Polarization Brightness - Log')
            ax5.patch.set(hatch='x', edgecolor='black')
            plt.tight_layout()

        if plotStats:
            fig1, (ax2, ax3) = plt.subplots(1,2,True, True)
            fig1.suptitle(self.batchName, verticalalignment = 'bottom')
            #import pdb
            #pdb.set_trace()
            fig1.set_size_inches(12,9)
            centroid = np.multiply(centroid, 3e5/self.env.lam0)
            sigma = self.std2V(sigma)
            

            centroid = np.ma.masked_invalid(centroid)
            throw = max(np.abs(np.nanmin(centroid)), np.abs(np.nanmax(centroid)))
            p2 = ax2.pcolormesh(self.zax, self.yax, centroid, cmap = 'RdBu', vmin = -throw, vmax = throw)
            ax2.patch.set(hatch='x', edgecolor='black')

            ax2.set_title('Centroid')
            ax2.set_ylabel(ystring)
            ax2.set_xlabel(zstring)
            cax1 = plt.colorbar(p2, ax = ax2)
            cax1.ax.set_title('km/s')

            sigma = np.ma.masked_invalid(sigma)
            p3 = ax3.pcolormesh(self.zax, self.yax, sigma, vmin = 25, vmax = 200)#, vmin = np.nanmin(sigma), vmax = np.nanmax(sigma)*0.8)
            ax3.patch.set(hatch='x', edgecolor='black')
            ax3.set_title('Line Width')
            ax3.set_xlabel(zstring)
            cax2 = plt.colorbar(p3, ax = ax3)
            cax2.ax.set_title('km/s')

            plt.tight_layout()

        plt.show()

        return
            #plt.colorbar(p0,cax=ax0)

    def coronagraph(self, array):
        """Normalize each radius so that there is better contrast"""
        #Set up impact bins
        zaxis = self.zax.tolist()
        yaxis = self.yax.tolist()
        mask = np.zeros_like(array, dtype=int)
        #self.rez = 50
        self.graphEdges = np.linspace(0.7*np.amin(zaxis), 1.2*np.amax(zaxis), self.corRez)
        #self.graphBins = np.zeros_like(self.graphEdges)
        self.graphList = [[] for x in np.arange(self.corRez)]
        self.graphList[0].append(0)

        yit = np.arange(len(yaxis))
        zit = np.arange(len(zaxis))

        #place each pixel intensity into an impact bin, and assign each pixel an impact
        for yy, yi in zip(yaxis, yit):
            for zz, zi in zip(zaxis, zit):
                r = np.sqrt(yy*yy + zz*zz)
                index = np.searchsorted(self.graphEdges, r)
                mask[yi,zi] = int(index)

                inten = array[yi,zi]
                self.graphList[index].append(inten)
                #self.graphBins[index] += inten

        #Get the statistics for each impact bin
        mins = []
        mids = []
        maxs = []
                ##TODO smooth these guys out
        for intensities in self.graphList:
            try:
                min = (np.nanmin(intensities))
                mid = (np.average(intensities))
                max = (np.nanmax(intensities))
            except:
                min, mid, max = np.nan, np.nan, np.nan
            mins.append(min)
            mids.append(mid)
            maxs.append(max)


            smins = ndimage.filters.gaussian_filter1d(mins, self.filt)
            smids = ndimage.filters.gaussian_filter1d(mids, self.filt)
            smaxs = ndimage.filters.gaussian_filter1d(maxs, self.filt)
            smins = ndimage.filters.gaussian_filter1d(smins, self.filt)
            smids = ndimage.filters.gaussian_filter1d(smids, self.filt)
            smaxs = ndimage.filters.gaussian_filter1d(smaxs, self.filt)

        #plt.plot(self.graphEdges,mins, 'b:')
        #plt.plot(self.graphEdges,mids, 'g:')
        #plt.plot(self.graphEdges,maxs, 'r:')

        #plt.plot(self.graphEdges,smins, 'b')
        #plt.plot(self.graphEdges,smids, 'g')
        #plt.plot(self.graphEdges,smaxs, 'r')
        
        #plt.show()

        if self.smooth:
            usemin = smins
            usemax = smaxs
        else:
            usemin = mins
            usemax = maxs

        #Create new scaled output array
        output = np.zeros_like(array)
        for yi in yit:
            for zi in zit:
                intensity = array[yi,zi]
                logint = (intensity)

                index = int(mask[yi,zi])
                inte = (logint-usemin[index]) / (usemax[index] - usemin[index])
                output[yi,zi] = inte
        #output[np.isnan(output)] = 0

        #plt.imshow(mask)
        #plt.show()
        
        return output




        #plt.pcolormesh(output)
        #plt.colorbar()
        #plt.show()



        #pdb.set_trace()
        #plt.plot(self.graphEdges, self.graphBins)
        #plt.show()
        #plt.pcolormesh(zaxis,yaxis,mask)
        #plt.show()






def pbRefinement(envsName, params, MIN, MAX, tol):
    #This function compares the pB at a height of zz with and without B, and finds the Bmin which minimizes the difference
    def runAtBminFull(params, Bmin):
        #This runs a multisim at a particular Bmin and returns the pB
        comm = MPI.COMM_WORLD
        root = comm.Get_rank() == 0
        simpoint.useB = True
        simpoint.g_Bmin = Bmin
        
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
        
        
        
        #From profileStats
                #return [power, mu, sigout, skew, kurt, perr, ratio]


        
            #error1 = np.sqrt(profNorm) + 1e-16 #, sigma = error2)
            #error2 = np.sqrt(profileRaw) + 1e-16 #, sigma=error1)
        #powerCon = poptCon[0] * np.sqrt(np.pi * 2 * poptCon[2]**2)
        ##Output only the asked for type.
        #if self.statType.casefold() in 'moment'.casefold():
        #    power, mu, sigmaStatFix, skew, kurt, amp, sigmaStat = powerM, muM, sigmaMFix, skewM, kurtM, ampM, sigmaM
        #elif self.statType.casefold() in 'gaussian'.casefold():
        #    power, mu, sigmaStatFix, skew, kurt, amp, sigmaStat = powerG, muG, sigmaGFix, skewM, kurtM, ampCon, sigmaG
        #else: raise Exception('Statistic Type Undefined')
        # * 1.030086 #This is the factor that makes it the same before and after psf
        
        #From makePSF
                     #import warnings
        #with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        #    import astropy.convolution as con
        #    self.con = con
        #if angSig is not None:
        #    diff = np.abs(self.lamAx[1] - self.lamAx[0])
        #    pix = int(np.ceil(angSig/diff))
        #    self.psf = self.con.Gaussian1DKernel(pix)
        #    #plt.plot(self.psf)
        #    #plt.show()
        #else: self.psf = None   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

def nothing():
    pass

