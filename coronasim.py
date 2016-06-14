# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:13:05 2016


@author: chgi7364
"""

print('Loading Dependencies...')
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
import skimage as ski
import sklearn.neighbors as nb
from collections import defaultdict
from skimage.feature import peak_local_max
import gridgen as grid
import progressBar as pb
import math
from astropy import units as u




np.seterr(invalid = 'ignore')

### Figure out BThresh

class simpoint:
    script_dir = os.path.dirname(os.path.abspath(__file__)) 

    rel_def_Bfile = '..\\dat\\mgram_iseed0033.sav'    
    def_Bfile = os.path.normpath(os.path.join(script_dir, rel_def_Bfile))

    rel_def_xiFile = '..\\dat\\xi_function.dat'    
    def_xiFile = os.path.normpath(os.path.join(script_dir, rel_def_xiFile))

    rel_def_bkFile = '..\\dat\\plasma_background.dat'    
    def_bkFile = os.path.normpath(os.path.join(script_dir, rel_def_bkFile))


    BMap = None
    xi_raw = None
    bk_dat = None
    
    rstar = 1
    B_thresh = 1.#6.0
    fmax = 8.2
    theta0 = 28

    mi = 1.6726219e-24 #grams per hydrogen
    #mi = 9.2732796e-23 #grams per Iron
    c = 2.998e10 #cm/second (base velocity unit is cm/s)
    KB = 1.380e-16 #ergs/Kelvin

    streamRand = np.random.RandomState()
    #thermRand = np.random.RandomState()
    
    def __init__(self, cPos = [0,0,1.5], findT = False, grid = None, Bfile = None, xiFile = None, bkFile = None):
 
        #Inputs
        self.cPos = cPos
        self.pPos = self.cart2sph(self.cPos)
        self.rx = self.r2rx(self.pPos[0])
        self.grid = grid

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

        #Load Plasma Background
        if bkFile is None: self.bkFile = simpoint.def_bkFile 
        else: self.bkFile = self.relPath(bkFile)
        if simpoint.bk_dat is None:
            x = np.loadtxt(self.bkFile, skiprows=10)
            simpoint.bk_dat = x
            simpoint.rx_raw = x[:,0]
            simpoint.rho_raw = x[:,1]
            simpoint.ur_raw = x[:,2]
            simpoint.vAlf_raw = x[:,3]
            simpoint.T_raw = x[:,4]

        #Initialization
        self.findTemp()
        self.findFootB()
        self.findDensity()
        self.findTwave(findT)
        self.findStreamIndex()
        self.findSpeeds()
        self.findVLOS(self.grid.ngrad)
        self.findIntensity()




  ## Temperature ######################################################################

    def findTemp(self):
        self.T = self.interp_rx_dat(simpoint.T_raw)




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
        #plt.imshow(simpoint.label_im, cmap = 'prism')
        #for co in coordinates:
        #    plt.scatter(*co)
        #plt.show()

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
            #mean_col = I[idx[:,0], idx[:,1]].mode(axis=0) #The pretty pictures part
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

    def findStreamIndex(self):
        self.streamIndex = self.label_im[self.find_nearest(self.BMap_x, self.foot_cPos[0])][
                                            self.find_nearest(self.BMap_y, self.foot_cPos[1])]

 

       
  ## Density ##########################################################################
 
    def findDensity(self):
        #Find the densities of the grid point
        self.densfac = self.findDensFac()
        self.rho = self.findRho()

        #self.num_den_min = self.minNumDense(self.rx)
        #self.num_den = self.densfac * self.num_den_min
        #self.rho_min = self.num2rho(self.num_den_min)
        #self.rho = self.num2rho(self.num_den)

    def findDensFac(self):
        # Find the density factor
        if np.abs(self.footB) < np.abs(self.B_thresh):
            return 1
        else:
            return (np.abs(self.footB) / self.B_thresh)**0.5

    def findRho(self):
        return self.interp_rx_dat(self.rho_raw) * self.densfac
        
    #def minNumDense(self, rx):
    #    #Find the number density floor
    #    return self.interp_rx_dat(self.num_dens_raw)
    #    #return 6.0e4 * (5796./rx**33.9 + 1500./rx**16. + 300./rx**8. + 25./rx**4. + 1./rx**2)
        
    #def num2rho(self, num_den):
    #    #Convert number density to physical units (cgs)
    #    return (1.841e-24) * num_den  
        
   


 
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
        #Find all of the time dependent velocities
        self.waveV = self.vRms*self.xi(t - self.twave + self.streamT)
        self.uTheta = self.waveV * np.sin(self.streamAngle)
        self.uPhi = self.waveV * np.cos(self.streamAngle)
        self.pU = [self.ur, self.uTheta, self.uPhi]
        self.cU = self.findCU()       
       
    def findUr(self):
        #Wind Velocity
        return self.interp_rx_dat(self.ur_raw) / self.densfac
        #return (1.7798e13) * self.fmax / (self.num_den * self.rx * self.rx * self.f)

    def findAlf(self):
        #Alfen Velocity
        return self.interp_rx_dat(self.vAlf_raw) / np.sqrt(self.densfac)
        #return 10.0 / (self.f * self.rx * self.rx * np.sqrt(4.*np.pi*self.rho))
    
    def findVPh(self):
        #Phase Velocity
        return self.vAlf + self.ur 
    
    def findvRms(self):
        #RMS Velocity
        self.S0 = 7.0e5
        return np.sqrt(self.S0*self.vAlf/((self.vPh)**2 * 
            self.rx**2*self.f*self.rho))

    def findVLOS(self, nGrad = None):
        if nGrad is not None: self.nGrad = nGrad
        self.vLOS = np.dot(self.nGrad, self.cU)
        return self.vLOS

    def findCU(self):
        #Finds the cartesian velocity components
        self.ux = -np.cos(self.pPos[2])*(self.ur*np.sin(self.pPos[1]) + self.uTheta*np.cos(self.pPos[1])) - np.sin(self.pPos[2])*self.uPhi
        self.uy = -np.sin(self.pPos[2])*(self.ur*np.sin(self.pPos[1]) + self.uTheta*np.cos(self.pPos[1])) - np.cos(self.pPos[2])*self.uPhi
        self.uz = self.ur*np.cos(self.pPos[1]) - self.uTheta*np.sin(self.pPos[1])
        return [self.ux, self.uy, self.uz]
    
    #def findRandU(self):
    #    #Wrong
    #    #amp = 0.01 * self.vRms * (self.rho / self.rho_min)**1.2
    #    #rand = simpoint.thermRand.standard_normal(3)
    #    return 0,0,0 #amp*rand

    #def findStreamU(self):
    #    #if self.streamIndex == 0: return 0.0, 0.0 #Velocities for the regions between streams
    #    #simpoint.streamRand.seed(int(self.streamIndex))
    #    #thisRand = simpoint.streamRand.standard_normal(2)
    #    ##Wrong scaling
    #    #streamTheta = 0.1*self.vRms*thisRand[0]
    #    #streamPhi = 0.1*self.vRms*thisRand[1]
    #    return 0,0 #streamTheta, streamPhi




  ## Time Dependence ##########################################################################    

    def findTwave(self, findT):
        #Finds the wave travel time to this point
        #Approximate Version
        twave_min = 161.4 * (self.rx**1.423 - 1.0)**0.702741
        self.twave_fit = twave_min * self.densfac
        
        if findT:
            #Real Version 
            radial = grid.sightline(self.foot_cPos, self.cPos)
            N = 10
            time = 0
            rLine = radial.cLine(N)
            for cPos in rLine:
                time += (1/simpoint(cPos, False, radial).vPh) / N
            self.twave = time * self.r2rx(radial.norm) * 69.63e9 #radius of sun in cm
        else:
            self.twave = self.twave_fit  
        self.twave_rat = self.twave/self.twave_fit
        

    def setTime(self,t):
        #Updates velocities to input time
        self.findTSpeeds(t)
        self.findVLOS()

    def xi(self, t):
        #Returns xi(t)
        if math.isnan(t):
            return math.nan
        else:
            t_int = int(t % simpoint.last_xi_t)
            xi1 = self.xi_raw[t_int]
            xi2 = self.xi_raw[t_int+1]
            return xi1+( (t%simpoint.last_xi_t) - t_int )*(xi2-xi1)




  ## Radiative Transfer ####################################################################

    def findIntensity(self, lam0 = 1000, lam = 1000):
        self.lam = lam #Angstroms
        self.lam0 = lam0 #Angstroms

        self.qt = 1

        self.lamLos =  self.vLOS * self.lam0 / simpoint.c
        self.deltaLam = self.lam0 / simpoint.c * np.sqrt(2 * simpoint.KB * self.T / simpoint.mi)
        self.lamPhi = 1/(self.deltaLam * np.sqrt(np.pi)) * np.exp(-((self.lam - self.lam0 - self.lamLos)/self.deltaLam)**2)
        self.intensity = self.rho**2 * self.qt * self.lamPhi * 1e32
        return self.intensity



        
  ## Misc Methods ##########################################################################

    def find_nearest(self,array,value):
        #Returns the index of the point most similar to a given value
        idx = (np.abs(array-value)).argmin()
        return idx

    def interp_rx_dat(self, array):
        #Interpolates an array(rx)
        if self.rx < 1. : return math.nan
        rxInd = int(self.find_nearest(simpoint.rx_raw, self.rx))
        val1 = array[rxInd]
        val2 = array[rxInd+1]
        slope = val2 - val1
        step = simpoint.rx_raw[rxInd+1] - simpoint.rx_raw[rxInd]
        discreteRx = simpoint.rx_raw[rxInd]
        diff = self.rx - discreteRx
        diffstep = diff / step
        return val1+ diffstep*(slope)

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



class simulate:

    def __init__(self, gridobj, N = None, iL = None, findT = False):
        print("Initializing Simulation...")
        self.grid  = gridobj
        #print(self.grid.ngrad)
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
        bar = pb.ProgressBar(self.Npoints)
        print("Beginning Simulation...")
        self.sPoints = []
        self.pData = []
        for cPos in self.cPoints:
            thisPoint = simpoint(cPos, self.findT, self.grid) 
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
        else: print('Bad Scaling')
        datSum = sum((v for v in scaleProp.ravel() if not math.isnan(v)))
        return scaleProp, datSum

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
   


    def Vars(self):
        #Returns the vars of the simpoints
        return self.sPoints[0].Vars()

    def Keys(self):
        #Returns the keys of the simpoints
        return self.sPoints[0].Vars().keys()


    ####################################################################
    ####################################################################
    def lineProfile(self):
        #Get a line profile at the current time
        intensity = np.zeros_like(self.lamAx)
        index = 0
        for lam in self.lamAx:
            for point in self.sPoints:
                intensity[index] += point.findIntensity(self.lam0, lam)
            index += 1
        return intensity

    def makeLamAxis(self, Ln = 100, lam0 = 1000, lamPm = 2):
        self.lamAx = np.linspace(lam0 - lamPm, lam0 + lamPm, Ln)
        return self.lamAx

    def evolveLine(self, t0 = 0, t1 = 1600, tn = 100, Ln = 100, lam0 = 1000, lamPm = 2):
        #Get the line profile over time and store in LineArray
        print('Timestepping...')
        
        self.lam0 = lam0
        self.times = np.linspace(t0, t1, tn)
        self.makeLamAxis(self, Ln, lam0, lamPm)
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

        self.lineList = self.lineArray.tolist()

        self.plotLineArray()
        self.fitGaussians()
        self.findMoments()

    def findMoments(self):
        #Find the moments of each line in lineList
        self.maxMoment = 3
        self.moment = []
        lineInd = 0
        for mm in np.arange(self.maxMoment):
            self.moment.append(np.zeros_like(self.times))

        for line in self.lineList:
            for mm in np.arange(self.maxMoment):
                self.moment[mm][lineInd] = np.dot(line, self.lamAx**mm)
            lineInd += 1

        self.findMomentStats()
        #self.plotMoments()

    def findMomentStats(self):
        self.power = self.moment[0]
        self.centroid = self.moment[1] / self.moment[0]
        self.sigma = np.sqrt(self.moment[2]/self.moment[0] - (self.moment[1]/self.moment[0])**2)
        self.plotMomentStats()

    def plotMomentStats(self):
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

    def plotMoments(self):
        f, axArray = plt.subplots(self.maxMoment, 1, sharex=True)
        mm = 0
        for ax in axArray:
            ax.plot(self.times, self.moment[mm])
            ax.set_title(str(mm)+" Moment")
            ax.set_ylabel('Angstroms')
            mm += 1
        ax.set_xlabel('Time (s)')
        plt.show(False)


    def plotLineArray(self):
        ## Plot the lineArray
        self.fig, ax = self.grid.plot(iL = self.iL)
        #print('')
        #print(np.shape(self.lineArray))
        im = ax.pcolormesh(self.lamAx.astype('float32'), self.times, self.lineArray)
        #ax.xaxis.get_major_formatter().set_powerlimits((0, 1))
        ax.set_xlabel('Angstroms')
        ax.set_ylabel('Time (s)')
        self.fig.subplots_adjust(right=0.89)
        cbar_ax = self.fig.add_axes([0.91, 0.10, 0.03, 0.8], autoscaley_on = True)
        self.fig.colorbar(im, cax=cbar_ax)
        grid.maximizePlot()
        plt.show(False)


    def peekLamTime(self, lam0 = 1000, lam = 1000, t = 0):
        intensity = np.zeros_like(self.sPoints)
        pInd = 0
        for point in self.sPoints:
            point.setTime(t)
            intensity[pInd] = point.findIntensity(lam0, lam)
            pInd += 1
        plt.plot(intensity)
        plt.show()

    def fitGaussians(self):
        self.amp = np.zeros_like(self.times)
        self.mu = np.zeros_like(self.times)
        self.std = np.zeros_like(self.times)
        self.area = np.zeros_like(self.times)

        def gauss_function(x, a, x0, sigma):
            return a*np.exp(-(x-x0)**2/(2*sigma**2))

        lInd = 0
        for line in self.lineList:
            sig0 = sum((self.lamAx - self.lam0)**2)/len(line)
            amp0 = np.max(line)
            popt, pcov = curve_fit(gauss_function, self.lamAx, line, p0 = [amp0, self.lam0, sig0])
            self.amp[lInd] = popt[0]
            self.mu[lInd] = popt[1] - self.lam0
            self.std[lInd] = popt[2] 
            self.area[lInd] = popt[0] * popt[2]  

            ## Plot each line fit
            #plt.plot(self.lamAx, gauss_function(self.lamAx, amp[lInd], mu[lInd], std[lInd]))
            #plt.plot(self.lamAx,  lineList[lInd])
            #plt.show()
            lInd += 1

        self.plotGaussStats()

    def plotGaussStats(self):
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







    #def gauss_function(x, a, x0, sigma):
    #    return a*np.exp(-(x-x0)**2/(2*sigma**2))




            #self.streamTheta, self.streamPhi = self.findStreamU() #All Zeros for now
            #self.randUr, self.randTheta, self.randPhi = self.findRandU() #All Zeros for now
            #self.uTheta = self.streamTheta + self.randTheta + 0.05* self.vRms*self.xi(t - self.twave)
            #self.uPhi = self.streamPhi + self.randPhi + 0.05*self.vRms*self.xi(t - self.twave) 



    ##def timeV(self, t0 = 0, t1 = 100, tstep = 2):
    ##    print('Timestepping...')
    ##    self.times = np.arange(t0, t1, tstep)
    ##    bar = pb.ProgressBar(self.Npoints*len(self.times))
    ##    stdList = []
    ##    vMeanList = []
    ##    for tt in self.times:
    ##        thisV = []
    ##        for point in self.sPoints:
    ##            point.setTime(tt)
    ##            thisV.append(point.gradV(self.grid.ngrad))
    ##            bar.increment()
    ##            bar.display()
    ##        stdList.append(np.std(np.array(thisV)))
    ##        vMeanList.append(np.mean(np.array(thisV)))           
    ##    bar.display(force = True)
    ##    self.vStd = np.array(stdList)
    ##    self.vStdErr = self.vStd / np.sqrt(len(stdList))
    ##    self.vMean = np.array(vMeanList)
    ##    self.timePlot()

    ##def timePlot(self):
    ##    #plt.semilogy(self.times, np.abs(self.vStd), self.times, np.abs(self.vSum))
    ##    plt.figure()
    ##    plt.plot(self.times, self.vMean)
    ##    plt.plot(self.times, self.vStdErr)
    ##    plt.title('mean and std of gradV over time')
    ##    plt.show(block = False)






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