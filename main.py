
import numpy as np
#import progressBar as pb
import matplotlib
#matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim
import time
import os
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = rank == 0
size = comm.Get_size()

if __name__ == '__main__':

    integration = 240

    #Environment Parameters
    envsName = 'Modern'
    fFileName = 'longfile'
    sim.environment.fFileName = fFileName
    maxEnvs = 6
    refineBmin = False
    calcFFiles = False
    processEnvironments = False

    #Batch Name
    batchName = 'nothing1' #inst'#'int{}h'.format(integration) #'timeRand' 'randLong' #'timeLong'#'rand'#'Waves' #"All" #"Wind" #"Thermal"

    # # # Which part of the program should run? # # #

    #Single Sim Playground
    simOne = False  

    #3D Stuff - ImageSim Parameters
    compute3d = False
    analyze3d = False

    NN3D = [200,200]
    sim.imagesim.N = (200, 4000)
    sim.imagesim.timeAx3D = np.arange(0, 120, 2)
    rez3D =  [1,1]
    target3D = [0,1.5]
    len3D = 20
    envInd = 0
    sim.imagesim.corRez = 2000
    sim.imagesim.filt = 10
    sim.imagesim.smooth = True

    #1D Stuff - ImpactSim Parameters
    compute = True  
    analyze = False  

    impactPoints = 3 
    iterations = 1

    b0 =  1.02
    b1 =  5#1.6 #46
    spacing = 'lin'

    N_line = (200,3000)
    rez = None #[3,3]
    size = [0.002, 0.01]
    timeAx = np.arange(0, 200, 2) #[int(x) for x in np.linspace(0,4000,15)] #np.arange(0,2000,2) #['rand'] #
    sim.simulate.randTime = False
    length = 10

    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #Simulation Properties
    sim.simpoint.useB = False
    sim.simpoint.g_useWaves = True   
    sim.simpoint.g_useWind = True

    sim.simpoint.matchExtrap = False
    sim.simpoint.wavesVsR = True
    sim.simpoint.useIonFrac = True
    sim.simpoint.g_useFluxAngle = True
    sim.simpoint.g_Bmin = 3.8905410
    sim.multisim.destroySims = True #This keeps memory from building up btwn multisims
    sim.multisim.keepAll = False
    sim.multisim.useMasters = False
    sim.batchjob.qcuts = [16,50,84]
    
    sim.batchjob.usePsf = True
    sim.batchjob.reconType = 'sub' #'Deconvolution' or 'Subtraction' or 'None'
    sim.batchjob.psfSig_FW = 0.06 #0.054 #angstroms FWHM


    printSim = False #This makes it show the generating profile progress bar
    widthPlot = True #Plot just line width instead of all 5 moments
    firstRun = True  #Overwrite any existing batch with this name

    #Run in parallel?

    parallel = True
    cores = 7

    ##Plotting Flags
    sim.simulate.plotSimProfs = False #Shows all the little gaussians added up
    
    sim.batchjob.plotFits = False #Plots the Different fits to the line w/ the raw line
    sim.batchjob.maxFitPlot = 1    

    sim.batchjob.hahnPlot = False #Plot the green Hahn Data on the primary plot
    sim.batchjob.plotRatio = False #Plot the ratio of the reconstruction/raw fits

    #Examine Batch Line Profiles
    showProfiles = False #Plot some number of line profiles at each impact parameter
    maxPlotLines = 30
    average = False
    norm = True 
    log = False

##################################################################################
##################################################################################

    #This header handles calling the program in parallel
    try: go = int(sys.argv[1])
    except: go = 1
    if parallel and go == 1 and (compute or refineBmin or compute3d):
        print("Starting MPI...")
        os.system("mpiexec -n {} python main.py 0".format(cores))
    else:
        

        ### Process Envs ###
        ####################
        if refineBmin:
            tol = 0.01
            MIN = 3
            MAX = 5
            b = 1.5
            iter = 1
            envs = sim.envrs(envsName).loadEnvs(maxEnvs)
            params = ["pbCalcs", envs, 1, iter, b, None, 600, rez, size, timeAx, length, False, False, False]
            useB = sim.simpoint.useB 
            sim.simpoint.Bmin = sim.pbRefinement(envsName, params, MIN, MAX, tol)
            sim.simpoint.useB = useB
        comm.barrier()

        if processEnvironments:
            if root:
                envrs1 = sim.envrs(envsName, fFileName)
                envs = envrs1.processEnvs(maxEnvs)
                #envrs1.showEnvs(maxEnvs)
                sys.stdout.flush()
        comm.barrier()
        if not calcFFiles: fFileName = None

        ### Level 3 ### BatchSim
        ########################

        if compute:
            envs = sim.envrs(envsName).loadEnvs(maxEnvs) 
            if firstRun:                   
                myBatch = sim.impactsim(batchName, envs, impactPoints, iterations, b0, b1, N_line, rez, size, timeAx, length, printSim, fName = fFileName, spacing = spacing)
            else:
                myBatch = sim.batch(batchName).restartBatch(envs)  
        if root:      
            if analyze:
                myBatch = sim.batch(batchName).analyzeBatch(widthPlot)
                #myBatch.fName='longfile'
                #myBatch.calcFfiles()
            if showProfiles: 
                try: myBatch
                except: myBatch = sim.batch(batchName).loadBatch()
                myBatch.plotProfiles(maxPlotLines)
                #myBatch.plotProfTogether(average, norm, log)

        if compute3d:
            envs = sim.envrs(envsName).loadEnvs(maxEnvs)[envInd]
            myBatch = sim.imagesim(batchName, envs, NN3D, rez3D, target3D, len3D)
        if analyze3d and root:
            try: myBatch
            except: myBatch = sim.batch(batchName).loadBatch()
            myBatch.plot()
        #### Level 1 ### Simulate ################################################################################################################################
        ################

        if simOne and root:
            print('Beginning...')
            df = grid.defGrid()
            env = sim.envrs(envsName).loadEnvs(100)[0]


            #sim.plotpB()

            #env.plot2DV()
            #if self.root: 
            #    import pdb 
            #    pdb.set_trace()
            #self.comm.barrier()

            #env.plot('ur_raw', 'rx_raw')
            x = 10
            y = 0.001
            z = 1.3

            position, target = [x, y, z], [-x, y, z]
            myLine = grid.sightline(position, target, coords = 'cart')
            #env.plotXi()
            #lineSim = sim.simulate(myLine, env, N = (200,4000), findT = True, getProf = True)
            ##lineSim.plot2('dangle', 'pPos', dim2 = 1)
            #lineSim.plot('dPB', linestyle = 'o', scaling = 'log')
            lineSim = sim.simulate(df.poleLine, env, N = 400, findT = False, getProf = False, printOut = True)
            
            #temps, _ = lineSim.get('T')
            #rad, _ = lineSim.get('pPos', dim = 0)
            #F = []
            ##temps = np.logspace(4,8,500)
            #for T in temps:
            #    #print(T)
            #    F.append(env.interp_frac(T))
            ##plt.plot(temps)
            #plt.plot(rad, F, label = 'Frac')
            ##plt.plot(rad, np.log10(temps)/18, label = 'Temp')

            #plt.yscale('log')
            ##plt.plot(10**env.chTemps,env.chFracs, label = 'Raw')
            #plt.legend()
            #plt.show()   

            #The cool new time evolution plots
            #T = 800
            #sim.simulate.movName = 'windowPlot.mp4'
            #lineSim.evolveLine(T,0,T)


            lineSim.plot('T', scaling = 'log', abscissa='pPos')
            lineSim.plot('frac', abscissa = 'pPos')
            #lineSim.plot('alfU2', cmap = 'RdBu', center = True, abscissa = 'pPos', absdim = 0)
            #lineSim.compare('uTheta', 'pU', p2Dim = 1, center = True)
            #lineSim.quiverPlot()




            ##lineSim.plot2('frac','nion', p1Scaling='log', p2Scaling='log')
            #lineSim.plotProfile()

            #bpoleSim = sim.simulate(df.bpolePlane, env, N = 200, findT = False, printOut = True)
            #bpoleSim.getProfile()
            #bpoleSim.plot('totalInt', cmap = 'brg', scaling = 'root', scale = 15)

        #position, target = [10, 3, 1.5], [-10, -3, 1.5]
        #cyline = grid.sightline(position, target, coords = 'Cart', rez = None, size = [0.002,0.01])
    
        #timeAx = [0] #np.arange(0,1500)
        #cylSim = sim.simulate(cyline, env, [1500, 3000], 1, False, True, timeAx = timeAx)
        #cylSim.getProfile()

        #cylSim.plot('densfac')
        #cylSim.plot2('vLOS','vLOSwind')
            #myBatch = sim.impactsim(*params, pBname = pBname)
            #if rank == 0:
            #    pBgoal = np.asarray(myBatch.pBavg, dtype = 'float')
            #else:
            #    pBgoal = np.empty(1, dtype = 'float')
            #comm.Bcast(pBgoal, root=0)

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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        