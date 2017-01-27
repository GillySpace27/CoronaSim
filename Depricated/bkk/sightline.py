# -*- coding: utf-8 -*-
"""
Created on Tue May 24 00:59:12 2016

@author: chgi7364
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import io
from scipy import interpolate as interp
import coronasim as sim
#import idlsave

class SightLine:
    default_step = 0.1
    
    def __init__(self, position, target, coords = 'Cart'):
#        print('Observer at {pos}, looking at {targ}'.format(pos = position, targ = target))

        self.look(position, target, coords)  
 
    def look(self, position, target, coords = 'Cart'):
#        Initialize the sight line between two points
        if coords == 'Cart':
            self.cPos = position
            self.cTarg = target
            self.pPos = self.cart2sph(position)
            self.pTarg = self.cart2sph(target)
        else:
            self.cPos = self.sph2cart(position)
            self.cTarg = self.sph2cart(target)
            self.pPos = position
            self.pTarg = target
            
        #Find the direction the line is pointing
        self.gradient = list(np.array(self.cTarg) - np.array(self.cPos))
        
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
        rho = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan(y/x)
        phi = np.arccos(z/rho)
        return [rho, theta, phi]
        
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        first = True
        for pos in self.cLine():
            if first: colr = 'red' 
            else: colr = 'blue'
            ax.scatter(*pos, color = colr)
            first = False
        #    print(myLine.cLine(jj))
            
        #Plot a sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        diameter = 1        
        
        x = diameter * np.outer(np.cos(u), np.sin(v))
        y = diameter * np.outer(np.sin(u), np.sin(v))
        z = diameter * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color='y', alpha = 0.5)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        axscale = 1.5*diameter
        ax.auto_scale_xyz([-axscale, axscale], 
                          [-axscale, axscale], [-axscale, axscale])
        
        plt.show()
        
    def cPoint(self, s):
        #Return the coordinates of a point along the line
        return (np.array(self.cPos) + np.array(self.gradient)*s).tolist()
        
    def pPoint(self, s):
        #Return the polar coordinates of a point along the line
        return self.cart2sph(self.cPoint(s))

    def cLine(self, step = None, smin=0, smax = 1):
        #Return the coordinates of the sightline
        if step == None: step = self.default_step
        line = []
        for ss in np.arange(smin, smax+step, step):
            line.append(self.cPoint(ss)) 
        return line
    
    def pLine(self, step = None, smin=0, smax = 1):  
        #Return the polar coordinates of the sightline
        if step == None: step = self.default_step
        line = []
        for ss in np.arange(smin, smax+step, step):
            line.append(self.pPoint(ss))           
        return line

        

        
if __name__ == "__main__":       
        
#    position, target = [0,-2,0], [0,2,0]
#    
#    myLine = SightLine(position, target)
    
    position, target = [2, 3*np.pi/4, 0.01], [2, -np.pi/4, 0.01]
    
    myLine = SightLine(position, target, "Sphere")    
    
    
#    print(myLine.pLine(0.2))
    myLine.plot()
    sim.simpoint(myLine.cPoint(0.8)).show()        
        
    
    #myLine.look(target, position)
    
#    line = myLine.cLine(0, 1, 5)
#    print(*line)
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
    
#    for jj in np.arange(0,1.1,0.1):
#        ax.scatter(*myLine.cLine(jj))
#    #    print(myLine.cLine(jj))
#    
#    plt.show()
    
    
    

#    property_names=[p for p in dir(SomeClass) if isinstance(getattr(SomeClass,p),property)]
#    print(property_names)
#    print(vars(myPoint))
#    myPoint.show()
    
    
    ## Next, plot density on a plane around the sun, 
    #add in integration for twave, and read in the B field stuff
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
