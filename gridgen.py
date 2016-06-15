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
import sys

zero = 1e-8

class generator:
    
    rstar = 1
    default_N = 100 #For plane, line is 1/10 * this
    
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
        
    def plotSphere(self, ax): 
        #Plot a sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        diameter = generator.rstar        
        
        x = diameter * np.outer(np.cos(u), np.sin(v))
        y = diameter * np.outer(np.sin(u), np.sin(v))
        z = diameter * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, rstride=4, cstride=4, color='y', alpha = 0.75)
        ax.set_xlabel("X axis")
        ax.set_ylabel("Y axis")
        ax.set_zlabel("Z axis")
        axscale = 1.5*diameter
        ax.auto_scale_xyz([-axscale, axscale], 
                          [-axscale, axscale], [-axscale, axscale])    

    def quadAx(self, quad):
        fig = plt.figure("CoronaSim")
        dataAxis = plt.subplot2grid((2,4), (0,2), colspan=2, rowspan = 2)
        if quad:
            ax1 = plt.subplot2grid((2,4), (0,0), projection = '3d', aspect = 'equal')
            ax1.view_init(elev=0., azim=0.)
            ax2 = plt.subplot2grid((2,4), (0,1), projection = '3d', aspect = 'equal')
            ax2.view_init(elev=90, azim=0.)
            ax3 = plt.subplot2grid((2,4), (1,0), projection = '3d', aspect = 'equal')
            ax3.view_init(elev=0, azim=90)
            ax4 = plt.subplot2grid((2,4), (1,1), projection = '3d', aspect = 'equal')
            quadAxis = [ax1, ax2, ax3, ax4]
        else:
            ax1 = plt.subplot2grid((2,4), (0,0), colspan=2, rowspan = 2, projection = '3d', aspect = 'equal')
            quadAxis = [ax1]

        return fig, dataAxis, quadAxis

    def plot(self, N = 10, iL = 1, show = False):
        #Plot the quadGrid with a sphere for the sun

        fig, dataAxis, quadAxis = self.quadAx(True)

        n = 0   
        if type(self) is sightline:      
            thisGrid = self.cLine(N = N, smax = iL)
            fig.suptitle('Sightline at Position = ({:0.2f}, {:0.2f}, {:0.2f}), \
                Target = ({:0.2f}, {:0.2f}, {:0.2f})'.format(*(self.cPos + self.cTarg)))
        elif type(self) is plane:
            thisGrid = self.cPlane(N = N, iL = iL)
            fig.suptitle('Plane with Normal = ({:0.2f}, {:0.2f}, {:0.2f}),\
                Offset = ({:0.2f}, {:0.2f}, {:0.2f})'.format(*(self.normal.tolist() + self.offset)))
        nax = 1
        for ax in quadAxis:
            for pos in thisGrid:
                if n < 2: colr = 'red' 
                else: colr = 'blue'
                ax.scatter(*pos, color = colr)
                n += 1
            ax.set_xlim([-1.5*iL, 1.5*iL])
            ax.set_ylim([-1.5*iL, 1.5*iL])
            ax.set_zlim([-1.5*iL, 1.5*iL])

            #self.ax.set_title()
            self.plotSphere(ax)
            if nax < 4:
                for tl in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
                    tl.set_visible(False) 
            n = 0 
            nax += 1   
        if show: 
            maximizePlot()
            plt.show()

        return fig, dataAxis

    def show(self):
        #Print all porperties and values
        myVars = vars(self)
        print("\nGenerator Properties")
        for ii in sorted(myVars.keys()):
            print(ii, " : ", myVars[ii])
            

class sightline(generator):
    def __init__(self, position = None, target = None, iL = 1, coords = 'Cart'):
        #print('Initializing Sightline, Observer at {pos}, looking at {targ}'.format(pos = position, targ = target))
        if position is None:
            position, target = [2, 1*np.pi/4, 1e-8], [2, -np.pi/4, 1e-8]
            coords = 'sphere'  
        self.iL = iL
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
        self.gradArr = np.asarray(self.gradient).astype(float)
        self.norm = np.linalg.norm(self.gradArr)  
        self.ngrad = self.gradArr / self.norm

    def cPoint(self, s):
        #Return the coordinates of a point along the line
        return (np.array(self.cPos) + self.gradArr*s).tolist()
        
    def pPoint(self, s):
        #Return the polar coordinates of a point along the line
        return self.cart2sph(self.cPoint(s))

    def cLine(self, N = None, smin=0, smax = 1):
        #Return the coordinates of the sightline
        if N is None: N = self.default_N/10
        line = []
        for ss in np.linspace(smin, smax, N):
            line.append(self.cPoint(ss)) 
        self.shape = [len(line), 1]  
        return line
    
    def pLine(self, N = None, smin=0, smax = 1):  
        #Return the polar coordinates of the sightline
        if N is None: N = self.default_N/10
        line = []
        for ss in np.linspace(smin, smax, N):
            line.append(self.pPoint(ss))  
        self.shape = [len(line), 1]         
        return line

   

class plane(generator):
    
    def __init__(self, normal = [0,1,0], offset = [0,3,-3], iL = 6, rotAxis = [-1,1,1], ncoords = 'Cart'):
        #print("Initializing Plane, normal = {}, offset = {}".format(normal, offset))

        self.iL = iL
        self.rotArray = np.asarray(rotAxis)
        if ncoords == 'Cart':
            self.normal = np.asarray(normal).astype(float)
            self.offset = offset
        else:
            self.normal = np.asarray(self.sph2cart(normal)).astype(float)
            if len(offset) == 3:
                self.offset = offset
            elif len(offset) == 2:
                self.offset = [normal[0], *offset]
        
        self.findGrads()

            
    def findGrads(self):
        #Determine the eigenvectors of the plane
        grad1 = np.cross(self.normal, self.rotArray)
        grad2 = np.cross(self.normal, grad1)
        
        self.ngrad1 = grad1 / np.linalg.norm(grad1)
        self.ngrad2 = grad2 / np.linalg.norm(grad2)
        self.ngrad = self.ngrad2
        self.nnormal = self.normal / np.linalg.norm(self.normal)
        self.noffset = self.nnormal*self.offset[0] + self.ngrad1*self.offset[1] + self.ngrad2*self.offset[2]

    def cPlane(self, N = None, iL = 1):
        #Return a list of points in the plane
        if N is None: N = self.default_N
        L = self.rstar*iL

        sGrad1 = self.ngrad1*L
        sGrad2 = self.ngrad2*L
        
        baseLine = sightline(sGrad1 + self.noffset, -sGrad1 + self.noffset).cLine(N)
        pos0 = baseLine[0]

        self.nx = len(baseLine)
        self.ny = len(sightline(pos0+sGrad2, pos0-sGrad2).cLine(N))
        self.shape = [self.nx, self.ny] 
       
        thisPlane = []
        for pos in baseLine:
           thisPlane.extend(sightline(pos+sGrad2, pos-sGrad2).cLine(N))
           
        return thisPlane

    def pPlane(self, N = None, iL = 1):
        #Return a list of points in the plane in polar Coords
        return [self.cart2sph(pos) for pos in self.cPlane(N, iL)]

class defGrid:

    def __init__(self):
        print('Generating Default Grids...')
        #Above the Pole        
        iL = 1
        normal1 = [0,0,1] 
        offset1 = [1.5, 0, 0]

        self.topPlane = plane(normal1, offset1, iL)

        #Slice of the Pole
        self.polePlane = plane()

        #Bigger Slice of the Pole
        self.bpolePlane = plane(iL = 8)

        #This line goes over the pole without touching it
        position, target = [2, np.pi/4, 0.01], [2, -np.pi/4, -0.01]
        self.primeLine = sightline(position, target, coords = 'sphere')

        #This line starts from north pole and goes out radially
        self.poleLine = sightline([1.1,0,0],[3.0,0,0], coords = 'Sphere')

    def impactLines(self, b0 = 1.05, b1= 1.5, N=5):
        lines = []
        x = 20
        y = 1e-8
        for zz in np.linspace(b0,b1,N):
            lines.append(sightline([x,y,zz], [-x,y,zz]))
        return lines
       
                        
def maximizePlot():
    try:
        mng = plt.get_current_fig_manager()
        backend = plt.get_backend()
        if backend == 'TkAgg':
            try:
                mng.window.state('zoomed')
            except:
                mng.resize(*mng.window.maxsize())
        elif backend == 'wxAgg':
            mng.frame.Maximize(True)
        elif backend[:2].upper() == 'QT':
            mng.window.showMaximized()
        else:
            return False
        return True
    except:
        return False        

        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
