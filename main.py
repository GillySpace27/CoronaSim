
import numpy as np
#import progressBar as pb
import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim
import time
from mpi4py import MPI


if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root = rank == 0
    envPath = '../dat/primeEnv'
    
    
    if root:
        print('\nCoronaSim!')
        print('Written by Chris Gilbert')
        print('-------------------------\n')
    #Simulate Common Grids
    df = grid.defGrid()
    #Initialize Simulation Environment
        # env = sim.environment()
        # env.save(envPath)

    # else:
        # df = None
        # envs = None

    # df = comm.bcast(df, root = 0)
    # envs = comm.bcast(envs, root = 0)

 
    ### Level 0 ### Simpoint 
    ###############
    #env = sim.envs('envs').loadEnvs(1)[0]
    #thisPoint = sim.simpoint(grid = df.bpolePlane, env = env) 
    #thisPoint.show()


    ### Level 1 ### Simulate
    ###############

    ## Misc Sims ##
    #topSim = sim.simulate(df.topPlane, N = 1000)
    #poleSim = sim.simulate(df.polePlane, findT = True, step = 0.01)

    ## bpoleSim ##
    #t = time.time()
    #bpoleSim = sim.simulate(df.bpolePlane, env[0], N = 750, findT = False, printOut = True)
    #t = time.time()
    #print('Elapsed Time: ' + str(time.time() - t))
    #bpoleSim.compare('rho', 'intensity', p1Scaling = 'log', p2Scaling = 'log')
    #bpoleSim.plot('t1', scaling = 'none')

    # env = sim.envs('envs').loadEnvs(1)[0]
    # print('Go')
    # lineSim = sim.simulate(df.impLine, env, N = 1500, findT = True)
    # lineSim.peekLamTime()

    
    #poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)
    #lineSim.plotProfile()
    #print(lineSim.getStats())

    #lineSim2 = sim.simulate(df.primeLine, env, N = (1000,10000), findT = True)
    #poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)
    #lineSim2.evolveLine(150)
    #lineSim2.setIntLam(200.01)
    #lineSim2.plot('intensity')
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

    ### Level 2 ### MultiSim
    ###############
    

    #lines = grid.rotLines(6)
    #lineSims = sim.multisim(lines, env, N = 200)
    #lineSims.plotLines()
    #plt.imshow(np.log(lineSims.getLineArray()))
    #plt.show()

    
    #lines = grid.rotLines()
    #lineSims = sim.multisim(lines, env, N = 1000)
    #plt.pcolormesh(np.log(lineSims.getLineArray()))
    #plt.show()


    ### Level 3 ### BatchSim
    ###############
    
#TODO make stats doable at the batch level

    #envs = sim.envs('Multienv').processEnvs()


    remote = False

    batchName = 'FinalLong' 
    
    if remote:

        envs = sim.envs('Multienv').loadEnvs()
        
        myBatch = sim.impactsim(envs, 40, 100)
        
        if root: myBatch.save(batchName)
    
    else:
        #env = sim.envs('Multienv').loadEnvs(1)[0]
        myBatch = sim.loadBatch(batchName)
        #myBatch.env = env
        myBatch.redoStats()
        #myBatch.plotStatsV()


   

    #if root:
    #    print('')
    #    print('Sim Name =')
    #    print([x for x in vars().keys() if "Sim" in x])
        
    #if root:
    #    print('')
    #    print('Batch Name =')
    #    print(batchName)     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        