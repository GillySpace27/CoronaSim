# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:13:05 2016


@author: chgi7364
"""

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
from scipy.optimize import curve_fit
import skimage as ski
import sklearn.neighbors as nb
from collections import defaultdict
from skimage.feature import peak_local_max
import gridgen as grid
import progressBar as pb
import math
from astropy import units as u

np.seterr(invalid = 'ignore')

### Make the time stepper more real
### Figure out BThresh

class simpoint:
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    rel_def_Bfile = '..\\dat\\mgram_iseed0033.sav'    
    def_Bfile = os.path.normpath(os.path.join(script_dir, rel_def_Bfile))
    rel_def_xiFile = '..\\dat\\xi_function.dat'    
    def_xiFile = os.path.normpath(os.path.join(script_dir, rel_def_xiFile))

    BMap = None
    xi_raw = None
    
    rstar = 1
    B_thresh = 1.0
    fmax = 8.2
    theta0 = 28

    mi = 1.6726219e-24 #grams per hydrogen
    c = 2.998e10 #cm/second (base velocity unit is cm/s)
    KB = 1.380e-16 #ergs/Kelvin

    streamRand = np.random.RandomState()
    thermRand = np.random.RandomState()
    
    def __init__(self, cPos = [0,0,1.5], findT = False, Bfile = None, xiFile = None):
 
        #Inputs
        self.cPos = cPos
        self.pPos = self.cart2sph(self.cPos)
        self.rx = self.r2rx(self.pPos[0])

        #Load Bmap
        if Bfile is None: self.Bfile = simpoint.def_Bfile 
        else: self.Bfile = self.relPath(Bfile)
        if simpoint.BMap is None: self.loadBMap()

        #Load Xi
        if xiFile is None: self.xiFile = simpoint.def_xiFile 
        else: self.xiFile = self.relPath(xiFile)
        if simpoint.xi_raw is None:
            x = np.loadtxt(self.xiFile, skiprows=1)
            simpoint.xi_t = x[:,0]
            simpoint.xi_raw = x[:,1]
            simpoint.last_xi_t = x[-1,0]

        #Initialization
        self.findFootB()
        self.findDensity()
        self.findTwave(findT)
        self.findStreamIndex()
        self.findSpeeds()


  ## Magnets ##########################################################################
        
    def loadBMap(self):
        Bobj = io.readsav(self.Bfile)
        simpoint.BMap_raw = Bobj.get('data_cap')
        #plt.imshow((np.abs(simpoint.BMap_raw)), cmap = "winter_r")
        #plt.colorbar()
        #plt.show()
        simpoint.BMap_x = Bobj.get('x_cap')
        simpoint.BMap_y = Bobj.get('y_cap')
        simpoint.BMap = interp.RectBivariateSpline(simpoint.BMap_x, simpoint.BMap_y, simpoint.BMap_raw)
        self.labelStreamers()

    def labelStreamers(self, thresh = 0.7):
        #Label the Streamers
        bdata = np.abs(simpoint.BMap_raw)
        validMask = bdata != 0

        blist = bdata.flatten().tolist()
        bmean =  np.mean([v for v in blist if v!=0])
        bmask = bdata > bmean * thresh
        label_im, nb_labels = ndimage.label(bmask)

        coord = ndimage.maximum_position(bdata, label_im, np.arange(1, nb_labels))
        coordinates = []
        for co in coord: coordinates.append(co[::-1])

        simpoint.label_im, simpoint.nb_labels = self.voronoify_sklearn(label_im, coordinates)
        simpoint.label_im *= validMask

    def voronoify_sklearn(self, I, seeds):
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
            #mean_col = I[idx[:,0], idx[:,1]].mode(axis=0)
            I2[idx[:,0], idx[:,1]] = label
        return I2, label

 
    def findfoot_Pos(self):

        #Find the footpoint of the field line
        Hfit  = 2.2*(self.fmax/10.)**0.62
        self.f     = self.fmax + (1.-self.fmax)*np.exp(-((self.rx-1.)/Hfit)**1.1)
        theta0_edge = self.theta0 * np.pi/180.
        theta_edge  = np.arccos(1. - self.f + self.f*np.cos(theta0_edge))
        edge_frac   = theta_edge/theta0_edge 
        coLat = self.pPos[1] /edge_frac
        self.foot_pPos = [self.rstar+1e-8, coLat, self.pPos[2]]
        self.foot_cPos = self.sph2cart(self.foot_pPos)
        
    def findFootB(self):
        #Find B
        self.findfoot_Pos()
        self.footB = self.BMap(self.foot_cPos[0], self.foot_cPos[1])[0][0]

    def find_nearest(self,array,value):
        idx = (np.abs(array-value)).argmin()
        return idx

    def findStreamIndex(self):
        self.streamIndex = self.label_im[self.find_nearest(self.BMap_x, self.foot_cPos[0])][
                                            self.find_nearest(self.BMap_y, self.foot_cPos[1])]

        
  ## Density ##########################################################################
 
    def findDensity(self):
        #Find the densities of the grid point
        self.num_den_min = self.minNumDense(self.rx)
        self.num_den = self.actualDensity(self.num_den_min)
        self.rho_min = self.num2rho(self.num_den_min)
        self.rho = self.num2rho(self.num_den)
        
    def minNumDense(self, rx):
        #Find the number density floor
        return 6.0e4 * (5796./rx**33.9 + 1500./rx**16. + 300./rx**8. + 25./rx**4. + 1./rx**2)
        
    def num2rho(self, num_den):
        #Convert number density to physical units (cgs)
        return (1.841e-24) * num_den  
        
    def actualDensity(self, density):
        # Increase density as function of B field
        if np.abs(self.footB) < np.abs(self.B_thresh):
            return density
        else:
            return density * (np.abs(self.footB) / self.B_thresh)**0.5   
 
  ## Velocity ##########################################################################

    def findSpeeds(self, t = 0):
        #Find all of the various velocities
        self.ur = self.findUr()
        self.vAlf = self.findAlf()
        self.vPh = self.findVPh()
        self.vRms = self.findvRms()

        simpoint.streamRand.seed(int(self.streamIndex))
        thisRand = simpoint.streamRand.random_sample(2)
        self.streamT =  thisRand[0] * simpoint.last_xi_t
        self.streamAngle = thisRand[1] * 2 * np.pi
        self.findTSpeeds(t)

    def findTSpeeds(self, t = 0):
        self.waveV = self.vRms*self.xi(t - self.twave + self.streamT)
        self.uTheta = self.waveV * np.sin(self.streamAngle)
        self.uPhi = self.waveV * np.cos(self.streamAngle)
        self.pU = [self.ur, self.uTheta, self.uPhi]
        self.cU = self.findCU()       
         
    def findUr(self):
        return (1.7798e13) * self.fmax / (self.num_den * self.rx * self.rx * self.f)

    def findAlf(self):
        return 10.0 / (self.f * self.rx * self.rx * np.sqrt(4.*np.pi*self.rho))
    
    def findVPh(self):
        return self.vAlf + self.ur 
    
    def findvRms(self):
        self.S0 = 7.0e5
        return np.sqrt(self.S0*self.vAlf/((self.vPh)**2 * 
            self.rx**2*self.f*self.rho))

    def findRandU(self):
        #Wrong
        #amp = 0.01 * self.vRms * (self.rho / self.rho_min)**1.2
        #rand = simpoint.thermRand.standard_normal(3)
        return 0,0,0 #amp*rand

    def findStreamU(self):
        #if self.streamIndex == 0: return 0.0, 0.0 #Velocities for the regions between streams
        #simpoint.streamRand.seed(int(self.streamIndex))
        #thisRand = simpoint.streamRand.standard_normal(2)
        ##Wrong scaling
        #streamTheta = 0.1*self.vRms*thisRand[0]
        #streamPhi = 0.1*self.vRms*thisRand[1]
        return 0,0 #streamTheta, streamPhi

    def findGradV(self, nGrad = None):
        if nGrad is not None: self.nGrad = nGrad
        self.vGrad = np.dot(self.nGrad, self.cU)
        return self.vGrad

    def findCU(self):
        #Finds the cartesian velocity components
        self.ux = -np.cos(self.pPos[2])*(self.ur*np.sin(self.pPos[1]) + self.uTheta*np.cos(self.pPos[1])) - np.sin(self.pPos[2])*self.uPhi
        self.uy = -np.sin(self.pPos[2])*(self.ur*np.sin(self.pPos[1]) + self.uTheta*np.cos(self.pPos[1])) - np.cos(self.pPos[2])*self.uPhi
        self.uz = self.ur*np.cos(self.pPos[1]) - self.uTheta*np.sin(self.pPos[1])
        return [self.ux, self.uy, self.uz]
    
  ## Time Dependence ##########################################################################    

    def findTwave(self, findT):
        #Approximate Version
        twave_min = 161.4 * (self.rx**1.423 - 1.0)**0.702741
        denFac = np.sqrt(self.num_den / self.num_den_min)
        self.twave_fit = twave_min * denFac
        
        if findT:
            #Real Version 
            radial = grid.sightline(self.foot_cPos, self.cPos)
            N = 10
            time = 0
            rLine = radial.cLine(N)
            for cPos in rLine:
                time += (1/simpoint(cPos, findT = False).vPh) / N
            self.twave = time * self.r2rx(radial.norm) * 69.63e9 #radius of sun in cm
        else:
            self.twave = self.twave_fit  
        self.twave_rat = self.twave/self.twave_fit
        

    def setTime(self,t):
        self.findTSpeeds(t)
        self.findGradV()

    def xi(self, t):
        if math.isnan(t):
            return math.nan
        else:
            t_int = int(t % simpoint.last_xi_t)
            xi1 = self.xi_raw[t_int]
            xi2 = self.xi_raw[t_int+1]
            return xi1+( (t%simpoint.last_xi_t) - t_int )*(xi2-xi1)

  ## Radiative Transfer ####################################################################

####################################################################
####################################################################

    def findIntensity(self, lam0 = 1000, lam = 1001):
        self.lam = lam #Angstroms
        self.lam0 = lam0 #Angstroms

        self.qt = 1
        self.T = 1e6 #Kelvin

        self.lamLos =  self.vGrad * self.lam0 / simpoint.c
        self.deltaLam = self.lam0 / simpoint.c * np.sqrt(2 * simpoint.KB * self.T / simpoint.mi)
        self.lamPhi = 1/(self.deltaLam * np.sqrt(np.pi)) * np.exp(-((self.lam - self.lam0 - self.lamLos)/self.deltaLam)**2)
        self.intensity = self.num_den**2 * self.qt * self.lamPhi * 1e-11
        return self.intensity

        
  ## Misc Methods ##########################################################################

    def r2rx(self, r):
        return r/self.rstar
                          
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
        
    def relPath(self, path):
        script_dir = os.path.dirname(os.path.abspath(__file__))   
        rel = os.path.join(script_dir, path)    
        return rel
            
    def show(self):
        #Print all porperties and values
        myVars = vars(self)
        print("\nSimpoint Properties")
        for ii in sorted(myVars.keys()):
            print(ii, " : ", myVars[ii])

    def Vars(self):
        return vars(self) 



class simulate:

    def __init__(self, gridobj, N = None, iL = None, findT = False):
        print("Initializing Simulation...")
        self.grid  = gridobj
        self.findT = findT
        if iL is None: self.iL = self.grid.iL
        else: self.iL = iL
        if N is None: self.N = self.grid.default_N/100
        else: self.N = N
        print("     Loading Grid...")
        if type(gridobj) is grid.sightline:
            self.cPoints = self.grid.cLine(self.N, smax = self.iL)

        elif type(gridobj) is grid.plane:
            self.cPoints = self.grid.cPlane(self.N, self.iL)
        else: 
            print("Invalid Grid")
            return

        self.Npoints = len(self.cPoints)
        self.shape = self.grid.shape
        self.shape2 = [*self.shape, -1]

        self.simulate_now()

    def simulate_now(self):
        print("Beginning Simulation...")
        bar = pb.ProgressBar(self.Npoints)

        self.sPoints = []
        self.pData = []
        for cPos in self.cPoints:
            thisPoint = simpoint(cPos, self.findT) 
            if type(self.grid) is grid.sightline:
                thisPoint.findGradV(self.grid.ngrad)
                thisPoint.findIntensity()   
            self.sPoints.append(thisPoint)
            self.pData.append(thisPoint.Vars())
            bar.increment()
            bar.display()
        bar.display(force = True)
        print('')
        print('')

    def get(self, myProperty, dim = None, scaling = 'None'):
        propp = np.array([x[myProperty] for x in self.pData])
        prop = propp.reshape(self.shape2)
        if not dim is None: prop = prop[:,:,dim]
        prop = prop.reshape(self.shape)
        if scaling == 'None':
            scaleProp = prop
        elif scaling == 'log':
            scaleProp = np.log10(prop)
        elif scaling == 'sqrt':
            scaleProp = np.sqrt(prop)
        datSum = sum((v for v in scaleProp.ravel() if not math.isnan(v)))
        return scaleProp, datSum

    def timeV(self, t0 = 0, t1 = 100, tstep = 2):
        print('Timestepping...')
        self.times = np.arange(t0, t1, tstep)
        bar = pb.ProgressBar(self.Npoints*len(self.times))
        stdList = []
        vMeanList = []
        for tt in self.times:
            thisV = []
            for point in self.sPoints:
                point.setTime(tt)
                thisV.append(point.gradV(self.grid.ngrad))
                bar.increment()
                bar.display()
            stdList.append(np.std(np.array(thisV)))
            vMeanList.append(np.mean(np.array(thisV)))           
        bar.display(force = True)
        self.vStd = np.array(stdList)
        self.vStdErr = self.vStd / np.sqrt(len(stdList))
        self.vMean = np.array(vMeanList)
        self.timePlot()

    def timePlot(self):
        #plt.semilogy(self.times, np.abs(self.vStd), self.times, np.abs(self.vSum))
        plt.figure()
        plt.plot(self.times, self.vMean)
        plt.plot(self.times, self.vStdErr)
        plt.title('mean and std of gradV over time')
        plt.show(block = False)


####################################################################
####################################################################


    def evolveLine(self, t0 = 0, t1 = 1000, tn = 500, Ln = 50, lam0 = 1000, lamPm = 2.5):

        print('Timestepping...')
        
        self.lam0 = lam0
        self.times = np.linspace(t0, t1, tn)
        self.lamAx = np.linspace(lam0 - lamPm, lam0 + lamPm, Ln)
        self.lineArray = np.zeros((tn, Ln))

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

        self.plotLineArray()
        self.fitGaussians()

    def lineProfile(self):
        intensity = np.zeros_like(self.lamAx)
        index = 0
        for lam in self.lamAx:
            for point in self.sPoints:
                intensity[index] += point.findIntensity(self.lam0, lam)
            index += 1
        return intensity

    def fitGaussians(self):
        lineList = self.lineArray.tolist()
        self.amp = np.zeros_like(self.times)
        self.mu = np.zeros_like(self.times)
        self.std = np.zeros_like(self.times)

        def gauss_function(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        lInd = 0
        for line in lineList:
            sig0 = sum((self.lamAx - self.lam0)**2)/len(line)
            amp0 = np.max(line)
            popt, pcov = curve_fit(gauss_function, self.lamAx, line, p0 = [amp0, self.lam0, sig0])
            self.amp[lInd] = popt[0]
            self.mu[lInd] = popt[1]
            self.std[lInd] = popt[2]            #####This is where I'm working
            ## Plot each line fit
            #plt.plot(self.lamAx, gauss_function(self.lamAx, amp[lInd], mu[lInd], std[lInd]))
            #plt.plot(self.lamAx,  lineList[lInd])
            #plt.show()
            lInd += 1

        self.plotLineStats()

    def plotLineStats(self):
        f, (ax1, ax2, ax3) = plt.subplots(3,1, sharex=True)
        ax1.plot(self.times, self.amp)
        ax1.set_title('Amplitude')
        ax2.plot(self.times, self.mu)
        ax2.set_title('Mean')
        ax2.set_ylabel('Angstroms')
        ax3.plot(self.times, self.std)
        ax3.set_title('Std')
        ax3.set_xlabel('Time (s)')
        plt.show()

    def plotLineArray(self):
        ## Plot the lineArray
        print('')
        print(np.shape(self.lineArray))
        plt.pcolormesh(self.lamAx.astype('float32'), self.times, self.lineArray)
        #ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        plt.xlabel('Angstroms')
        plt.ylabel('Time (s)')
        plt.show()




    def peekLamTime(self, lam0 = 1000, lam = 1000, t = 0):
        intensity = np.zeros_like(self.sPoints)
        pInd = 0
        for point in self.sPoints:
            point.setTime(t)
            intensity[pInd] = point.findIntensity(lam0, lam)
            pInd += 1
        plt.plot(intensity)
        plt.show()

        


            




    def plot(self, property, dim = None, scaling = 'None'):
        scaleProp, datSum = self.get(property, dim, scaling)
        self.fig, ax = self.grid.plot(iL = self.iL)
 
        if type(self.grid) is grid.sightline:
            #Line Plot
            im = ax.plot(scaleProp)
            datSum = datSum / self.N
        elif type(self.grid) is grid.plane:
            #Image Plot
            im = ax.imshow(scaleProp, interpolation='none')
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

    def Vars(self):
        #Returns the vars of the simpoints
        return self.sPoints[0].Vars()

    def Keys(self):
        #Returns the keys of the simpoints
        return self.sPoints[0].Vars().keys()

class defGrid:

    def __init__(self):
        #Above the Pole        
        iL = 1
        normal1 = [0,0,1] 
        offset1 = [1.5, 0, 0]

        self.topPlane = grid.plane(normal1, offset1, iL)

        #Slice of the Pole
        self.polePlane = grid.plane()

        #Bigger Slice of the Pole
        self.bpolePlane = grid.plane(iL = 8)

        #This line goes over the pole without touching it
        position, target = [2, np.pi/4, 0.01], [2, -np.pi/4, -0.01]
        self.primeLine = grid.sightline(position, target, coords = 'sphere')

        #This line starts from north pole and goes out radially
        self.poleLine = grid.sightline([1.1,0,0],[3.0,0,0], coords = 'Sphere')

    def gauss_function(x, a, x0, sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))




            #self.streamTheta, self.streamPhi = self.findStreamU() #All Zeros for now
            #self.randUr, self.randTheta, self.randPhi = self.findRandU() #All Zeros for now
            #self.uTheta = self.streamTheta + self.randTheta + 0.05* self.vRms*self.xi(t - self.twave)
            #self.uPhi = self.streamPhi + self.randPhi + 0.05*self.vRms*self.xi(t - self.twave) 










        ##Label Contiguous regions
        #this_label_im, this_nb_labels = ndimage.label(smBmask)

        #plt.imshow(this_label_im, cmap = "winter_r")
        #plt.colorbar()
        #plt.show()

        
        #for ii in weights:
        #    passInd += 1
        #    #Mask only pixels greater than weighted mean
        #    bmean =  np.mean([v for v in bdata.tolist() if v!=0])
        #    bMask = bdata > bmean*ii
        #    #Label Contiguous regions
        #    this_label_im, this_nb_labels = ndimage.label(bMask)
        #    #Append to previous labels
        #    this_label_im += simpoint.nb_labels
        #    this_label_im *= bMask
        #    #Record passInd
        #    simpoint.bPassInd += passInd * bMask
        #    #Record Labels
        #    simpoint.label_im += this_label_im
        #    simpoint.nb_labels += this_nb_labels
        #    simpoint.bN.append(this_nb_labels)
        #    #plt.imshow(np.log10(this_label_im), cmap = 'prism')
        #    #plt.colorbar()
        #    #plt.show()
        #    bdata *= ~bMask










    #def getPointSpeeds(self, pPos):
    #    #Find all of the velocities for an arbitrary point
    #    u = self.findU(pPos)
    #    vAlf = self.findAlf(pPos)
    #    vPh = self.u + self.vAlf
    #    return vPh, u, vAlf
    
    #def getPointDensity(self, pPos):
    #    #Find the density of another grid point
    #    rx = self.r2rx(pPos[0])
    #    B = self.findB(pPos, self.BMap)
    #    num_den_min = self.minNumDense(rx)
    #    num_den = self.actualDensity(num_den_min, B)
    #    return self.num2rho(num_den), num_den