
import numpy as np
#import progressBar as pb
import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim
import time


if __name__ == '__main__':

    print('CoronaSim!')
    print('Written by Chris Gilbert')
    print('-------------------------\n')

    ##Simulate Common Grids
    df = grid.defGrid()
 
   ### Level 0 ### Simpoint 
    ###############

    #thisPoint = sim.simPoint(grid = df.bpolePlane) 


    ### Level 1 ### Simulate
    ###############

    ## Misc Sims ##
    #topSim = sim.simulate(df.topPlane, N = 1000)
    #poleSim = sim.simulate(df.polePlane, findT = True, step = 0.01)

    ## bpoleSim ##
    #t = time.time()
    #bpoleSim = sim.simulate(df.bpolePlane, N = 1000, findT = False)
    #t = time.time()
    #print('Elapsed Time: ' + str(time.time() - t))
    ##bpoleSim.compare('rho', 'intensity', p1Scaling = 'log', p2Scaling = 'log')
    #bpoleSim.plot('twave')


    ### Level 2 ### CoronaSim
    ###############
    

    lines = df.impactLines(50)
    env = sim.environment()
    impactSims = sim.coronasim(lines, env, N = 1000)

    


    #lineSim = sim.simulate(df.primeLine, N = 1e3, findT = True)
    #poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)
    #lineSim.plot('densfac')
    #print(lineSim.getStats())

    #lineSim2 = sim.simulate(df.primeLine, N = 2e3, findT = True)
    ##poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)
    ##lineSim.plot('densfac')
    #print(lineSim2.getStats())
    ## Useful Functions
        #mysim.plot('property')
        #mysim.Keys()
        #mysim.timeV(t0,t1,step)

    #topSim.plot('uPhi')
    #bpoleSim.plot('uPhi')
    #poleLineSim.plot('vGrad')
    #lineSim.evolveLine()
    #lineSim.timeV(t1 = 2000, step = 3)
    #bpoleSim.plot('T')
    #sim.simpoint()


    #The whole code in one line. WOO!
    #mySim = sim.simulate(sim.defGrid().bpolePlane, step = 0.1)#.plot('rho', scale = 'log')








    #    np.savetxt('density.txt', denss)
    #import matplotlib
    #matplotlib.use('tkagg')

    print('')
    print('Sim Name =')
    print([x for x in vars().keys() if "Sim" in x])