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

#Defines the Grid on which to simulate
class generator:
    
    rstar = 1

    backflag = True
    
    def __iter__(self):
        return self
    
    def back(self):
        self.currS -= self.step

    def incStep(self, mult):
        if self.step < self.maxStep:
            self.step = min(self.step * mult, self.maxStep)

    def set2minStep(self):
        self.step = self.minStep
        
    def setMinStep(self, step):
        self.minStep = step
        
    def setMaxStep(self, step):
        self.maxStep = step
        
    def setStep(self, step):
        self.step = step


        
    def decStep(self, mult):
        if self.step > self.minStep:
            self.step = max(self.step / mult, self.minStep)

    def reset(self):
        self.currS = 0
        self.step = maxStep
    
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

    def quadAx(self, quad = True):
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

    def quadAxOnly(self):
        fig = plt.figure("CoronaSim")

        ax1 = plt.subplot2grid((2,2), (0,0), projection = '3d', aspect = 'equal')
        ax1.view_init(elev=0., azim=0.)
        ax2 = plt.subplot2grid((2,2), (0,1), projection = '3d', aspect = 'equal')
        ax2.view_init(elev=90, azim=0.)
        ax3 = plt.subplot2grid((2,2), (1,0), projection = '3d', aspect = 'equal')
        ax3.view_init(elev=0, azim=90)
        ax4 = plt.subplot2grid((2,2), (1,1), projection = '3d', aspect = 'equal')
        quadAxis = [ax1, ax2, ax3, ax4]
        dataAxis = []
        return fig, dataAxis, quadAxis

    def plot(self, iL = 1, show = False, axes = None):
        #Plot the quadGrid with a sphere for the sun

        if axes is None: fig, dataAxis, quadAxis = self.quadAx()
        else: fig, dataAxis, quadAxis = axes

        n = 0   
        if type(self) is sightline:      
            thisGrid = self.cGrid(N = 25, iL = iL)
            fig.suptitle('Sightline at Position = ({:0.2f}, {:0.2f}, {:0.2f}), \
                Target = ({:0.2f}, {:0.2f}, {:0.2f})'.format(*(self.cPos + self.cTarg)))
        elif type(self) is plane:
            thisGrid = self.cGrid(N = 10, iL = iL)
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

            
#A line of points between two given points
class sightline(generator):

    default_N = 1000

    def __init__(self, position = None, target = None, 
            iL = 1, coords = 'Cart', findT = True, rez = None, size = None):
        #print('Initializing Sightline, Observer at {pos}, looking at {targ}'.format(pos = position, targ = target))
        if position is None:
            position, target = [2, 1*np.pi/4, 1e-8], [2, -np.pi/4, 1e-8]
            coords = 'sphere'  
        self.iL = iL
        self.findT = findT
        self.maxStep = 1/1500
        self.minStep = 1/10000
        self.coords = coords
        self.currLine = 0
        self.look(position, target, coords)
        self.stepChange = 4
        if rez is not None:
            self.makeLineList(rez, size)
 
    def look(self, position, target, coords = 'Cart'):
#        Initialize the sight line between two points
        self.currS = 0
        self.step = self.maxStep
        if coords.lower() == 'cart':
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

    def makeLineList(self, rez, size):
        self.NLines = rez[0] * rez[1]
        rotAxis = [0,0,1]
        self.currS = 1e8
        self.startPoints = plane(self.ngrad, self.cPos, 0.5, rotAxis, absolute = True).cGrid(rez[0], size[0], rez[1], size[1])
        self.endPoints = plane(self.ngrad, self.cTarg, 0.5, rotAxis, absolute = True).cGrid(rez[0], size[0], rez[1], size[1])

    def cPoint(self, s):
        #Return the coordinates of a point along the line
        return (np.array(self.cPos) + self.gradArr*s).tolist()
        
    def pPoint(self, s):
        #Return the polar coordinates of a point along the line
        return self.cart2sph(self.cPoint(s))

    def cGrid(self, N = None, smin=0, iL = 1):
        #Return the coordinates of the sightline
        if N is None: N = self.default_N
        line = []
        try:
            for start, end in zip(self.startPoints, self.endPoints):
                self.look(start, end)
                for ss in np.linspace(smin, iL, N):
                    line.append(self.cPoint(ss)) 
        except:
            for ss in np.linspace(smin, iL, N):
                line.append(self.cPoint(ss)) 
            
        self.shape = [len(line), 1]  
        return line
    
    def pGrid(self, N = None, smin=0, iL = 1):  
        #Return the polar coordinates of the sightline
        if N is None: N = self.default_N
        line = []
        for ss in np.linspace(smin, iL, N):
            line.append(self.pPoint(ss))  
        self.shape = [len(line), 1]         
        return line
        
    def setN(self, N):
        self.step = 1/N

    def reset(self):
        self.currS = 0
        self.step = self.maxStep

    def __next__(self):
        if self.currS > 1:
            try: 
                start = self.startPoints[self.currLine]
                end = self.endPoints[self.currLine]
                self.look(start, end)
                self.currLine += 1
            except:
                raise StopIteration
        pt = self.cPoint(self.currS)
        self.currS += self.step
        return (pt, self.step)


#TODO create cylinder

#A plane normal to a given vector
class plane(generator):
    #TODO Make plane adaptive
    default_N = 1000

    def __init__(self, normal = [0,1,0], offset = [0,3,-3], iL = 6, rotAxis = [-1,1,1], ncoords = 'Cart', findT = False, absolute = False):
        #print("Initializing Plane, normal = {}, offset = {}".format(normal, offset))
        self.absolute = absolute
        self.findT = findT
        self.iL = iL
        self.rotArray = np.asarray(rotAxis)
        if ncoords.lower() == 'cart':
            self.normal = np.asarray(normal).astype(float)
            self.offset = offset
        else:
            self.normal = np.asarray(self.sph2cart(normal)).astype(float)
            if len(offset) == 3:
                self.offset = offset
            elif len(offset) == 2:
                self.offset = [normal[0], offset[0], offset[1]]
        
        self.findGrads()

        
    def __next__(self):
        if not self.defGrid:
            raise StopIteration
        return (self.defGrid.pop(0), 1/self.N)

    def setN(self, N):
        self.defGrid = self.cGrid(N = N, iL = self.iL)
        
    def findGrads(self):
        #Determine the eigenvectors of the plane
        grad1 = np.cross(self.normal, self.rotArray)
        grad2 = np.cross(self.normal, grad1)
        
        self.ngrad1 = grad1 / np.linalg.norm(grad1)
        self.ngrad2 = grad2 / np.linalg.norm(grad2)
        self.ngrad = self.ngrad2
        self.nnormal = self.normal / np.linalg.norm(self.normal)
        if self.absolute:
            self.noffset = self.offset 
        else:
            self.noffset = self.nnormal*self.offset[0] + self.ngrad1*self.offset[1] + self.ngrad2*self.offset[2]

    def cGrid(self, N = None, iL = 1, N2 = None, iL2 = None):
        #Return a list of points in the plane
        if N is None: self.N = self.default_N
        else: self.N = N
        if N2 is None: self.N2 = self.N
        else: self.N2 = N2
        if iL2 is None: iL2 = iL

        L1 = self.rstar*iL
        L2 = self.rstar*iL2

        sGrad1 = self.ngrad1*L1
        sGrad2 = self.ngrad2*L2

        baseLine = sightline(sGrad1 + self.noffset, -sGrad1 + self.noffset).cGrid(self.N)
        pos0 = baseLine[0]
        self.nx = len(baseLine)
        self.ny = len(sightline(pos0+sGrad2, pos0-sGrad2).cGrid(self.N2))
        self.shape = [self.nx, self.ny]
        self.Npoints = self.nx * self.ny
        thisPlane = []
        for pos in baseLine:
           thisPlane.extend(sightline(pos+sGrad2, pos-sGrad2).cGrid(self.N2))
        return thisPlane

    def pGrid(self, N = None, iL = 1, N2 = None):
        #Return a list of points in the plane in polar Coords
        if N is None: self.N = self.default_N
        else: self.N = N
        if N2 is None: self.N2 = self.N
        else: self.N2 = N2
        return [self.cart2sph(pos) for pos in self.cGrid(N, iL, N2)]


#Generates default grids
class defGrid:

    def __init__(self):
        # print('Generating Default Grids...')
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
        position, target = [2, np.pi/4, 0.001], [2, -np.pi/4, -0.001]
        self.primeLine = sightline(position, target, coords = 'sphere')

        #This line goes over the pole without touching it
        position, target = [5, 0.001, 1.5], [-5, 0.001, 1.5]
        self.primeLineLong = sightline(position, target, coords = 'cart')

        #This line goes over the pole without touching it
        position, target = [10, 0.001, 1.5], [-10, 0.001, 1.5]
        self.primeLineVLong = sightline(position, target, coords = 'cart')

        #This line starts from north pole and goes out radially
        self.poleLine = sightline([1.1,0,0],[3.0,0,0], coords = 'Sphere')

        b = 1.03
        self.impLine = sightline([5,1e-8,b],[-5,1e-8,b], findT = True)


def impactLines(N=5, b0 = 1.05, b1= 1.5, len = 50):
    #Generate lines with a fixed angle but varying impact parameter
    lines = []
    x = len
    y = 1e-8
    #bax = np.logspace(np.log10(b0),np.log10(b1),N)
    bax = np.linspace(b0,b1,N)
    for zz in bax:
        lines.append(sightline([x,y,zz], [-x,y,zz], findT = True))
    #List of grids, list of labels
    return [lines, bax] 

def rotLines(N = 20, b = 1.05, offset = 0, x0 = 5, rez = None, size = None):
    #Generate lines with a fixed impact parameter but varying angle
    lines = []
    y0 = 1e-8
    angles = np.linspace(0, np.pi*(1 - 1/N), N)
    for theta in angles:
        theta += offset
        x = x0 * np.sin(theta) + y0 * np.cos(theta)
        y = x0 * np.cos(theta) - y0 * np.sin(theta)
        thisLine = sightline([x,y,b], [-x,-y,b], findT = True, rez = rez, size = size)
        #thisLine.plot(show = True)
        lines.append(thisLine)
    return [lines, angles]      

def image(N = [20,20], rez=[0.5,0.5], target = [0,1.5], len = 10):
    yy = np.linspace(target[0] - rez[0]/2, target[0] + rez[0]/2, N[0])
    zz = np.linspace(target[1] - rez[1]/2, target[1] + rez[1]/2, N[1])
    lines = []
    coords = []
    yi = 0
    ii = 0
    for y in yy:
        zi = 0
        for z in zz:
            line = sightline([len,y,z],[-len,y,z])
            line.findT = False
            line.index = (yi,zi,ii)
            lines.append(line)
            coords.append((y,z))
            zi += 1
            ii += 1
        yi += 1

    return [[lines, coords]], [yy,zz]
                        
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




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
