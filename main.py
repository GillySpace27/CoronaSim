
import numpy as np
#import progressBar as pb
import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim
import time
from mpi4py import MPI
import sys

if __name__ == '__main__':

    remote = False
    firstRun = False
    redoStats = False

    batchName = 'test'
    impactPoints = 20
    iterations = 1
    b0 = 1.05
    b1 = 1.5

    envsName = 'smoothEnvs'
    maxEnvs = 10
   
    ### Level 3 ### BatchSim
    ###############
    
    #envs = sim.envs('smoothEnvs').processEnvs()
    if False:
        if remote:
            if firstRun:       
                envs = sim.envs(envsName).loadEnvs(maxEnvs)
                myBatch = sim.impactsim(batchName, envs, impactPoints, iterations, b0, b1)
            else:
                myBatch = sim.restartBatch(batchName)        
        else:
            myBatch = sim.plotBatch(batchName, redoStats)




        #myBatch = sim.loadBatch(batchName)
        #myBatch.redoStats()

        #env = sim.envs('Multienv').loadEnvs(1)[0]
        #myBatch.env = env
        #myBatch.plotStatsV()

    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #size = comm.Get_size()
    #root = rank == 0

    #sys.stdout.flush()

    #Simulate Common Grids
    df = grid.defGrid()



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



    ### Level 1 ### Simulate
    ###############

    env = sim.envs(envsName).loadEnvs(1)[0]

    ## Misc Sims ##
    #topSim = sim.simulate(df.topPlane, N = 1000)
    #poleSim = sim.simulate(df.polePlane, findT = True, step = 0.01)

    ## bpoleSim ##
    #t = time.time()
    bpoleSim = sim.simulate(df.bpolePlane, env, N = 750, findT = False, printOut = True)
    #t = time.time()
    #print('Elapsed Time: ' + str(time.time() - t))
    #bpoleSim.compare('rho', 'intensity', p1Scaling = 'log', p2Scaling = 'log')
    bpoleSim.plot('streamIndex', scaling = 'none')


    # print('Go')
    # lineSim = sim.simulate(df.impLine, env, N = 1500, findT = True)
    # lineSim.peekLamTime()

    
    #poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)
    #lineSim.plotProfile()
    #print(lineSim.getStats())
    #poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)

    #env = sim.envs('smoothEnvs').loadEnvs(1)[0]
    #lineSim2 = sim.simulate(df.primeLine, env, N = (1000,10000), findT = True)
    #lineSim2.evolveLine(200, 0, 200)

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


    ### Level 0 ### Simpoint 
    ###############
    #env = sim.envs('envs').loadEnvs(1)[0]
    #thisPoint = sim.simpoint(grid = df.bpolePlane, env = env) 
    #thisPoint.show()


   

  




  
 

    #if root:
    #    print('')
    #    print('Sim Name =')
    #    print([x for x in vars().keys() if "Sim" in x])
        
    #if root:
    #    print('')
    #    print('Batch Name =')
    #    print(batchName)     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        