
import numpy as np
#import progressBar as pb
import matplotlib
#matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

    #Environment Parameters
    envsName = 'ionFreeze2'
    fFileName = 'hq'
    sim.environment.fFileName = fFileName
    
    refineBmin = False
    processEnvironments = True
    calcFFiles = False #Turn off B
    

    #Batch Name
    batchName = 'ionFreeze'#'ionFreeze' #inst'#'int{}h'.format(integration) #'timeRand' 'randLong' #'timeLong'#'rand'#'Waves' #"All" #"Wind" #"Thermal"

    # # # Which part of the program should run? # # #

    #Single Sim Playground
    simOne = False 

    #1D Stuff - ImpactSim Parameters
    compute = False
    analyze = False 
    
    try: #Plotflags
        sim.batchjob.pMass = True #Plot with temperature and non-thermal velocity from fits
        sim.batchjob.pIon = False #Plot with just the binned widths for all ions
        sim.batchjob.pMFit = False #Plot with straight fit lines on the ions
        sim.batchjob.pWidth = False #Plot with the hist velocities for each of the elements on its own plot
        sim.batchjob.pPB = False #Plot the polarization brightness
        sim.batchjob.pProportion = False #Plot the 4 ways of looking at the model parameters
        sim.batchjob.plotIon = 1
        sim.batchjob.pIntRat = 'save'  #Plot the intensities and fraction CvR # set = to 'save' to save images

        #For statistics:
        sim.batchjob.resonant = True #Use resonantly scattered light 
        sim.batchjob.collisional = True #Use collisionally excited light
    except: pass

    impactPoints = 10
    iterations = 1

    maxEnvs = 1
    sim.environment.maxIons = 3
    timeAx = [0]#np.arange(0, 400) #[int(x) for x in np.linspace(0,4000,15)] #np.arange(0,2000,2) #['rand'] #

    b0 =  1.05
    b1 =  5 #6 #1.6 #46
    spacing = 'lin'

    N_line = (600,3000)
    sim.batchjob.autoN = True
    rez = None #[3,3]
    size = [0.002, 0.01]
    sim.simulate.randTime = False
    length = 10
    

    #3D Stuff - ImageSim Parameters
    compute3d = False
    analyze3d = False

    NN3D = [100,100]
    sim.imagesim.N = 600
    sim.imagesim.timeAx3D = [0]#np.arange(0, 120, 2)
    rez3D =  [1,1]
    target3D = [0,1.5]
    len3D = 10
    envInd = 0
    sim.imagesim.corRez = 1000
    sim.imagesim.filt = 1
    sim.imagesim.smooth = True

    # # # # # # # # # # # # # # # # # # # # # # # # # #
    #Simulation Properties
    sim.simpoint.useB = False
    sim.simpoint.g_useWaves = False   
    sim.simpoint.g_useWind = True

    sim.simpoint.voroB = True #Use the voronoi average instead of the raw Bmap
    sim.simpoint.matchExtrap = True
    sim.simpoint.wavesVsR = True
    sim.simpoint.g_useFluxAngle = True
    sim.simpoint.useIonFrac = True #Use Ionization Fractions
    sim.environment.ionFreeze = False #Freeze ionization states at freezing height

    sim.multisim.destroySims = True #This keeps memory from building up btwn multisims
    sim.multisim.keepAll = False
    sim.multisim.useMasters = False
    
    sim.batchjob.usePsf = True
    sim.batchjob.reconType = 'sub' #'Deconvolution' or 'Subtraction' or 'None'
    sim.batchjob.redoStats = True
    sim.batchjob.plotbinFits = False #Plots the binned and the non-binned lines, and their fits, during stats only
    sim.batchjob.plotheight = 1

    printSim = False #This makes it show the generating profile progress bar
    firstRun = True  #Overwrite any existing batch with this name

    #Run in parallel?

    parallel = False
    cores = 7

    ##Plotting Flags
    sim.simulate.plotSimProfs = False #Shows all the little gaussians added up
    
    sim.batchjob.plotFits = False #Plots the Different fits to the line w/ the raw line
    sim.batchjob.maxFitPlot = 10    

    

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

    #All the call code is hidden
    #This header handles calling the program in parallel
    try: 
        go = int(sys.argv[1])
    except: go = 1
    if parallel and go == 1 and (compute or refineBmin or compute3d):
        print("Starting MPI...")
        os.system("mpiexec -n {} python main.py 0".format(cores))
        print("Parallel Job Complete")
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
                myBatch = sim.batch(batchName).analyzeBatch()
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
    #All the call code is hidden

        if simOne and root:
            print('Beginning...')
            df = grid.defGrid()
            env = sim.envrs(envsName).loadEnvs(100)[0]
            #env.fPlot()
            #env.plot2DV()

            ###xx = np.linspace(0,300,100)
            ###x0s = [125, 150, 175]
            ###sigma = 25
            ###a = [0.5, 500]
            ###a2 = [0.5, 1, 0.5]
            ###colors = ['r', 'b']
            ###from scipy.optimize import curve_fit

            ###def gauss_function(x, a, x0, sigma): return a*np.exp(-(x-x0)**2/(2*sigma**2))

            ###ans = []
            ###for amp, color in zip(a, colors):
            ###    array = np.zeros_like(xx)
            ###    for x0, a in zip(x0s, a2):
            ###        gauss = gauss_function(xx, amp*a, x0, sigma)
            ###        array += gauss
            ###        plt.plot(xx, gauss, color = color)
            ###    plt.plot(xx, array, '--', color = color)
            ###    poptRaw, pcovRaw = curve_fit(gauss_function, xx, array, p0 = [1,150,30])  
            ###    sigmaRaw = np.abs(poptRaw[2])
            ###    ans.append(sigmaRaw)
            ###print(ans)
            ###plt.show()

            
            if True:
                #Plot a sightline
                y = 0.001
                x = 20 
                z = 1.01
                N = 2000

                position, target = [x,y, z], [-x, y, z]
                myLine = grid.sightline(position, target, coords = 'cart')
                lineSim = sim.simulate(myLine, env, N = N, findT = True, getProf = True, printOut=True)
                #plt.xlim((0.000012382, 0.000012396))
                #plt.yscale('log')
                #plt.show()
                #profR = lineSim.profilesR[0]
                #lamax = np.squeeze(lineSim.ions[0]['lamAx'])
                #plt.plot(lamax, profR, 'k', lw=3)
                ##plt.yscale('log')
                #plt.show()           

                ions = [2]
                
                ax = lineSim.plot(['totalIntC','totalIntR'], yscale='log', frame = False, show = True, abscissa= 'cPos', ion = ions, savename=z)
                
                #r = 1
                #for ii in ions:
                #    int = np.max(lineSim.ions[ii]['I0array'])
                #    plt.plot(r, int, 'o', label = "SUMER Amplitude")
                
                
                #plt.show()
                #lineSim.plot(['radialV', 'ur'], ion = -1, abscissa = 'cPos', frame = False)
            #lineSim.plot('N', ion = -1, abscissa = 'cPos', yscale = 'log', norm = True)
            #lineSim.plot('delta', abscissa='cPos')
            #print(env.interp_f1(3.5))
            #lineSim.plot('uw', abscissa = 'pPos')

            #vlos = lineSim.get('vLOS')
            #plt.hist(vlos, 200)
            #plt.xlabel('Velocity')
            #plt.ylabel('Counts')
            #plt.title('Velocity Distribution along LOS')
            #plt.show()

            #twave = lineSim.get('twave')
            #r = lineSim.get('pPos', dim=0)
            
            #ans = env.interp(r, twave, 3)
            #print(ans)

            #sim.plotpB()

            #ex = 4

            #xx = np.linspace(0,1,60)
            #kk = xx**ex

            #kmin = 1.0001
            #kmax = 6
            #dif = kmax - kmin
            #qq = kk * dif + kmin
            

            #env.plot2DV()
            #if self.root: 
            #    import pdb 
            #    pdb.set_trace()
            #self.comm.barrier()

            #env.plot('ur_raw', 'rx_raw')

            ##For plotting the buildline plots:
            #y = 0.001
            #x = 20 
            #z = 2
            #N = 302

            #position, target = [x, y, z], [-x, y, z]
            #myLine = grid.sightline(position, target, coords = 'cart')
            #lineSim = sim.simulate(myLine, env, N = N, findT = True, getProf = True)


            ##This bit here is to plot something at many impacts on the same plot
            #y = 0.001
            #x = 20 
            #z = 2
            #N = 302
            #zlist = np.linspace(1.01,5,6).tolist()

            #p1 = 'urProjRel'
            #p2 = 'rmsProjRel'


            #z = zlist.pop(0)
            #position, target = [x, y, z], [-x, y, z]
            #myLine = grid.sightline(position, target, coords = 'cart')
            #lineSim = sim.simulate(myLine, env, N = N, findT = True, getProf = True)
            #ax1 = lineSim.plot(p1, refresh = True, frame = False, show = False, label = z, abscissa = 'cPos', axes = 1e-9)
            #ax2 = lineSim.plot(p2, refresh = True, frame = False, show = False, label = z, abscissa = 'cPos', axes = 1e-9)

            #for z in zlist:
            #    color = next(ax1._get_lines.prop_cycler)['color']
            #    position, target = [x, y, z], [-x, y, z]
            #    myLine = grid.sightline(position, target, coords = 'cart')
            #    lineSim = sim.simulate(myLine, env, N = N, findT = True, getProf = True)
            #    lineSim.plot(p1, refresh = True, useax = ax1, show = False, label = z, c = color, abscissa = 'cPos', axes = False)
            #    lineSim.plot(p2, refresh = True, useax = ax2, show = False, label = z, c = color, abscissa = 'cPos', axes = False)
            #ax1.legend()
            #ax2.legend()
            #plt.show()



            ##lineSim.plotProfile()
            #lineSim.plot('totalInt', ion = -1, frame = False, abscissa='pPos', yscale = 'log')
            #ax = lineSim.plot('frac', ion = -1, yscale = 'log', frame = False, abscissa = 'pPos', ylim = [1e-5,1], xlim = [1,4],show = False)

            #if sim.environment.ionFreeze and True:
            #    ax.set_color_cycle(None)
            #    for ion in env.ions:
            #        color = next(ax._get_lines.prop_cycler)['color']
            #        R = ion['r_freeze']
            #        T = env.interp_T(R)
            #        frac = env.interp_frac(T, ion)
            #        ax.plot(R, frac, color = color, marker = 'o')
            #        print("{}_{}: {}".format(ion['ionString'],ion['ion'], ion['r_freeze']))


            ##lineSim.plot2('dangle', 'pPos', dim2 = 1)
            #lineSim.plot('dPB', linestyle = 'o', scaling = 'log')

            #env.fPlot()
            if False:
                #Plot a plane
                lineSim = sim.simulate(df.polePlane, env, N = 50, findT = False, getProf = False, printOut = True)
                lineSim.plot('streamIndex', cmap='prism', threeD=False, sun=True)
                
                #lineSim.plot('streamIndex', sun = True) #, ion = -1, abscissa = 'T', yscale = 'log')
            #lineSim.plot('densfac')

            ###ax = lineSim.plot('rho', scaling = 'log', frame = False, vmax = -16, vmin = -21, cmap = 'inferno', clabel = 'log($g/cm^2$)', suptitle = 'Mass Density', extend = 'both')
            #ax2 = lineSim.plot('alfU1', cmap = 'RdBu', center = True, clabel='km/s', show = False)
            #streams = lineSim.get('streamIndex')
            #env.plotEdges(ax2, streams, False)
            #plt.show()

            #ax2 = lineSim.plot('pU', dim = 1, cmap = 'RdBu', center = True, clabel='km/s')


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


            #lineSim.plot('T', scaling = 'log', abscissa='pPos')
            #lineSim.plot('frac', abscissa = 'pPos')
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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        