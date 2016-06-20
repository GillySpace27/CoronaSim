
import numpy as np
#import progressBar as pb
import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim
import time
from mpi4py import MPI


if __name__ == '__main__':
    use_MPI = True

    if use_MPI:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        root = rank == 0
    else: root = True

    if root:
        print('\nCoronaSim!')
        print('Written by Chris Gilbert')
        print('-------------------------\n')
        ##Simulate Common Grids
        df = grid.defGrid()
        ##Initialize Simulation Environment
        env = sim.environment()
    else:
        df = None
        env = None

    if use_MPI:
        df = comm.bcast(df, root = 0)
        env = comm.bcast(env, root = 0)
 
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
    #bpoleSim = sim.simulate(df.bpolePlane, env, N = 1000, findT = False)
    #t = time.time()
    #print('Elapsed Time: ' + str(time.time() - t))
    ##bpoleSim.compare('rho', 'intensity', p1Scaling = 'log', p2Scaling = 'log')
    #bpoleSim.compare('alfV1', 'alfV2')


    ### Level 2 ### CoronaSim
    ###############
    

    lines = grid.impactLines(400)
    impactSims = sim.coronasim(lines, env, N = 2000, use_MPI = use_MPI)

    


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
    if root:
        print('')
        print('Sim Name =')
        print([x for x in vars().keys() if "Sim" in x])