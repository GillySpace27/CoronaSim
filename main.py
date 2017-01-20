
import numpy as np
#import progressBar as pb
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim
import time
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = rank == 0
size = comm.Get_size()

if __name__ == '__main__':

    #Environment Parameters
    envsName = 'envs-chianti'
    maxEnvs = 1

    #Which part of the program should run?
    mainProgram = True
    try: #Main program Parameters
        remote = True 
        firstRun = True
        mainplot = True
        redoStats = False
    except: pass
    simOne = False
    processEnvironments = False

    #Batch Parameters
    batchName = 'test'
    impactPoints = 5
    iterations = 3
    b0 = 1.05
    b1 = 1.5

    N_line = (1000, 3000)
    rez = None #[3,3]
    size = [0.002, 0.01]
    timeAx = [0] #np.arange(0,1500)
    printSim = False #This makes it show the generating profile progress bar
    
    #Examine Batch Line Profiles
    showProfiles = True
    maxPlotLines = 3

##################################################################################
##################################################################################

    ### Process Envs ###
    if processEnvironments:
        envrs1 = sim.envrs(envsName)
        envs = envrs1.processEnvs(maxEnvs)
        #envrs1.showEnvs(maxEnvs)

    ### Level 3 ### BatchSim
    ###############

    if mainProgram:
        if remote:
            if firstRun:       
                envs = sim.envrs(envsName).loadEnvs(maxEnvs)
                myBatch = sim.impactsim(batchName, envs, impactPoints, iterations, b0, b1, N_line, rez, size, timeAx, printSim)
            else:
                myBatch = sim.batch(batchName).restartBatch()        
        if root and mainplot:
            myBatch = sim.batch(batchName).plotBatch(redoStats)
            if showProfiles: myBatch.plotProfiles(maxPlotLines)


    #### Level 1 ### Simulate
    ################

    if simOne:
        print('Beginning...')
        df = grid.defGrid()
        env = sim.envrs(envsName).loadEnvs(1)[0]
        
        lineSim = sim.simulate(df.impLine, env, N = 1500, findT = True)
        lineSim.plot2('frac','nion', p1Scaling='log', p2Scaling='log')
        lineSim.plotProfile()

        #bpoleSim = sim.simulate(df.bpolePlane, env, N = 500, findT = False, printOut = True)
        ##bpoleSim.getProfile()
        #bpoleSim.plot('frac', cmap = 'brg')

    #position, target = [10, 3, 1.5], [-10, -3, 1.5]
    #cyline = grid.sightline(position, target, coords = 'Cart', rez = None, size = [0.002,0.01])
    
    #timeAx = [0] #np.arange(0,1500)
    #cylSim = sim.simulate(cyline, env, [1500, 3000], 1, False, True, timeAx = timeAx)
    #cylSim.getProfile()

    #cylSim.plot('densfac')
    #cylSim.plot2('vLOS','vLOSwind')



    #df = grid.defGrid()

    #env = sim.envs(envsName).loadEnvs(1)[0]

    #bpoleSim = sim.simulate(df.bpolePlane, env, N = 500, findT = False, printOut = True)
    #bpoleSim.setTime()
    #bpoleSim.plot('alfU1', cmap = 'RdBu', center = True)

    #bpoleSim.plot('vLOS', scaling = 'none', cmap = 'RdBu' )

    #lineSim2 = sim.simulate(df.primeLine, env, N = (1000,10000), findT = True)
    #print((lineSim2.Npoints))
    #lineSim2.evolveLine(200, 0, 200)

    ## Misc Sims ##
    #topSim = sim.simulate(df.topPlane, N = 1000)
    #poleSim = sim.simulate(df.polePlane, findT = True, step = 0.01)

    ## bpoleSim ##
    #t = time.time()

    #t = time.time()
    #print('Elapsed Time: ' + str(time.time() - t))
    #bpoleSim.compare('rho', 'intensity', p1Scaling = 'log', p2Scaling = 'log')



    # print('Go')
    # lineSim = sim.simulate(df.impLine, env, N = 1500, findT = True)
    # lineSim.peekLamTime()

    
    #poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)
    #lineSim.plotProfile()
    #print(lineSim.getStats())
    #poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)

    #env = sim.envs('smoothEnvs').loadEnvs(1)[0]


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


    #comm = MPI.COMM_WORLD
    #rank = comm.Get_rank()
    #size = comm.Get_size()
    #root = rank == 0

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


    ### Level 0 ### Simpoint 
    ###############
    #env = sim.envs(envsName).loadEnvs(1)[0]
    #df = grid.defGrid()
    #thisPoint = sim.simpoint(grid = df.bpolePlane, env = env) 
    #thisPoint.setTime()
    #thisPoint.show()


   

  




  
 

    #if root:
    #    print('')
    #    print('Sim Name =')
    #    print([x for x in vars().keys() if "Sim" in x])
        
    #if root:
    #    print('')
    #    print('Batch Name =')
    #    print(batchName)     
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        