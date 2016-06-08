
import numpy as np
#import progressBar as pb
import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim

##Creates an object with some common grids
df = sim.defGrid()

##Simulate Common Grids
#topSim = sim.simulate(df.topPlane, N = 1000)
#poleSim = sim.simulate(df.polePlane, findT = True, step = 0.01)
#bpoleSim = sim.simulate(df.bpolePlane, N = 500, findT = False)

#lineSim = sim.simulate(df.primeLine, N = 1500, findT = True)
poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)

## Useful Functions
    #mysim.plot('property')
    #mysim.Keys()
    #mysim.timeV(t0,t1,step)

#topSim.plot('uPhi')
#bpoleSim.plot('ur')

poleLineSim.plot('ur')
#lineSim.timeV(t1 = 2000, step = 3)
#lineSim.plot('twave')
#sim.simpoint()


#The whole code in one line. WOO!
#mySim = sim.simulate(sim.defGrid().bpolePlane, step = 0.1)#.plot('rho', scale = 'log')



#myPoint = sim.simpoint()

#bmap = myPoint.BMap_raw


#xpoints = bmap.x_cap
#ypoints = bmap.y_cap
#plt.imshow(bdata)
#plt.show()
#print(type(bdata))

#sx = ndimage.sobel(bdata, axis = 0, mode  = 'constant')
#sy = ndimage.sobel(bdata, axis = 1, mode  = 'constant')
#sob = np.hypot(sx, sy)
#plt.imshow((sob))
#plt.show()




#    np.savetxt('density.txt', denss)
#import matplotlib
#matplotlib.use('tkagg')

    
