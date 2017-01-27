# -*- coding: utf-8 -*-
"""
Created on Wed May 25 19:13:05 2016

@author: chgi7364
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import io
from scipy import interpolate as interp


class GridPoint:
    
    rel_def_Bfile = '..\\dat\\mgram_iseed0033.sav'    
    script_dir = os.path.dirname(os.path.abspath(__file__))   
    def_Bfile = os.path.normpath(os.path.join(script_dir, rel_def_Bfile))
    BMap = None
    BMap_raw = None
    
    def __init__(self, cPos, findT = True, Bfile = None):
        #Parameters
        self.rstar = 1
        self.B_thresh = 1.0
        self.fmax = 8.2
        self.theta0 = 28
        
        #Inputs
        self.cPos = cPos
        self.pPos = self.cart2sph(self.cPos)
        self.rx = self.r2rx(self.pPos[0])

        #File IO
        if Bfile is None: self.Bfile = GridPoint.def_Bfile 
        else: self.Bfile = self.relPath(Bfile)
        if GridPoint.BMap is None: 
            GridPoint.BMap_raw = io.readsav(self.Bfile)
            GridPoint.BMap = interp.RectBivariateSpline(GridPoint.BMap_raw.x_cap, GridPoint.BMap_raw.y_cap, GridPoint.BMap_raw.data_cap)

        #Initialization
        self.findB()
        self.findDensity()
        self.findSpeeds()
        if findT: self.findTwave()
            
            
    def findfoot_pPos(self):
        #Find the footpoint of the field line
        Hfit  = 2.2*(self.fmax/10.)**0.62
        self.f     = self.fmax + (1.-self.fmax)*np.exp(-((self.rx-1.)/Hfit)**1.1)
        theta0_edge = self.theta0 * np.pi/180.
        theta_edge  = np.arccos(1. - self.f + self.f*np.cos(theta0_edge))
        edge_frac   = theta_edge/theta0_edge 
        coLat = self.pPos[1] /edge_frac
        self.foot_pPos = [self.rstar, coLat, self.pPos[2]]
        self.foot_cPos = self.sph2cart(self.foot_pPos)
        
    def findB(self):
        self.findfoot_pPos()
        self.B = self.BMap(self.foot_cPos[0], self.foot_cPos[1])[0][0]
        

        
    def findDensity(self):
        #Find the densities of the grid point
        self.num_den_min = self.minNumDense(self.rx)
        self.num_den = self.actualDensity(self.num_den_min)
        self.rho_min = self.num2rho(self.num_den_min)
        self.rho = self.num2rho(self.num_den)
        
    def findSpeeds(self):
        #Find all of the various velocities
        self.u = self.findU()
        self.vAlf = self.findAlf()
        self.vPh = self.findVPh()
        self.vrms = self.findVRms()
        
    def minNumDense(self, rx):
        #Find the number density floor
        return 6.0e4 * (5796./rx**33.9 + 1500./rx**16. + 
            300./rx**8. + 25./rx**4. + 1./rx**2)
        
    def num2rho(self, num_den):
        #Convert number density to physical units (cgs)
        return (1.841e-24) * num_den  
        
    def actualDensity(self, density):
        # Increase density as function of B field
        if self.B < self.B_thresh:
            return density
        else:
            return density * (self.B / self.B_thresh)**0.5           
            
    def getPointSpeeds(self, pPos):
        #Find all of the velocities for an arbitrary point
        u = self.findU(pPos)
        vAlf = self.findAlf(pPos)
        vPh = self.u + self.vAlf
        return vPh, u, vAlf
    
    def getPointDensity(self, pPos):
        #Find the density of another grid point
        rx = self.r2rx(pPos[0])
        B = self.findB(pPos, self.BMap)
        num_den_min = self.minNumDense(rx)
        num_den = self.actualDensity(num_den_min, B)
        return self.num2rho(num_den), num_den
        
        
    def findAlf(self):
        return 10.0 / (self.f * self.rx * self.rx * np.sqrt(4.*np.pi*self.rho))
        
    def findU(self):
        return (1.7798e13) * self.fmax / (self.num_den * self.rx * self.rx * self.f)
    
    def findVPh(self):
        return self.vAlf + self.u 
    
    def findVRms(self):
        self.S0 = 7.0e5
        return np.sqrt(self.S0*self.vAlf/((self.u+self.vAlf)**2 * 
            self.rx*self.rx*self.f*self.rho))
    
    
    def findTwave(self):
        #Approximate Version
        twave_min = 161.4 * (self.rx**1.423 - 1.0)**0.702741
        denFac = np.sqrt(self.num_den / self.num_den_min)
        self.twave_fit = twave_min * denFac
        
        #Real Version
        radial = SightLine(self.foot_pPos, self.pPos, "Sphere")
        step = 0.1
        time = 0
        rLine = radial.cLine(step)
        for cPos in rLine:
            time = time + (1/GridPoint(cPos, findT = False).vPh) * step #####THIS NEEDS TO BE FINDU
        self.twave = time    
                
    def r2rx(self, r):
        return r/self.rstar
                          
    def sph2cart(self, sph):
        #Change coordinate systems
        rho, theta, phi = sph[:]
        x = np.array(rho)*np.cos(np.array(theta))*np.sin(np.array(phi))
        y = np.array(rho)*np.sin(np.array(theta))*np.sin(np.array(phi))
        z = np.array(rho)*np.cos(np.array(phi))
        return [x, y, z]
        
    def cart2sph(self, cart):  
        #Change coordinate systems
        x,y,z = cart[:]
        rho = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan(y/x)
        phi = np.arccos(z/rho)
        return [rho, theta, phi]
        
    def relPath(self, path):
        script_dir = os.path.dirname(os.path.abspath(__file__))   
        rel = os.path.join(script_dir, path)    
        return rel
            
    def show(self):
        #Print all porperties and values
        myVars = vars(self)
        print("\nGridpoint Properties")
        for ii in sorted(myVars.keys()):
            print(ii, " : ", myVars[ii])