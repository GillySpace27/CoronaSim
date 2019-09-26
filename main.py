
import numpy as np
import os
import matplotlib
if os.name == 'nt': matplotlib.use('qt5agg', force=True)
import matplotlib.pyplot as plt
plt.ioff()

import gridgen as grid
import coronasim as sim
from mpi4py import MPI
import sys
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
root = rank == 0
size = comm.Get_size()

sim.simulate.vectorize = False

if __name__ == '__main__':
    if size > 1 and root: print('{} Workers Opened'.format(size-1))

    # Environment Parameters
    envsName = 'Remastered'
    sim.environment.fFile = 'Remastered2'
    sim.environment.fFile_lin = 'Remastered_lin2'
    sim.environment.weightPower = 2

    processEnvironments = False
    storeF = False #'new_sq' # This is the name to store the f files to
    refineBmin = False

    # Batch Name
    batchName = 'Projection2'
    params = sim.runParameters(batchName)
    params.firstRun(True)  # Overwrite?

    params.useB(False)
    params.useWaves(False)
    params.useWind(True)
    params.windFactor(1)
    params.doChop(False) # Cut out the continuum of the incident light
    params.makeLight(True)

    # # # Which part of the program should run? # # # #

    # Single Sim Playground
    simOne = True

    # 1D Stuff - ImpactSim Parameters
    compute = False
    analyze = False
    sim.batchjob.redoStats = True

    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Compute Properties

    confirm = False
    # params.resolution(500)
    params.impacts(1.01,11,100)

    # How many lines should it do at each point?
    lines = 1
    # params.maxIons(100)
    # params.lamPrimeRez(500)
    # params.lamRez(200)

    # Run in parallel?
    sim.batchjob.usePool = False
    useMPI = False
    cores = 3


    # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Analyze Settings

    sim.batchjob.pMass = True  # Show temperature measurements
    sim.batchjob.pMass2 = False  # Show Moran Measurements
    sim.batchjob.pIon = False  # Plot with just the binned widths for all ions
    sim.batchjob.pMFit = False  # Plot with straight fit lines on the ions
    sim.batchjob.pWidth = False  # 'save' # 'save'  # Plot with the velocity statistics for each of the elements on its own plot
    sim.batchjob.plotBkHist = False  # Plot the hists in the background of the pWidth
    sim.batchjob.pPB = False  # Plot the polarization brightness
    sim.batchjob.pProportion = False  # Plot the 4 ways of looking at the model parameters
    sim.batchjob.plotIon = 1
    sim.batchjob.pIntRat = False  # 'save'  #Plot the intensities and fraction CvR # set = to 'save' to save images
    sim.batchjob.plotF = False
    sim.batchjob.showInfo = True

    # For statistics: Remember to turn on redostats if you change this
    sim.batchjob.resonant = True  # Use resonantly scattered light
    sim.batchjob.collisional = True  # Use collisionally excited light

    sim.batchjob.usePsf = False
    sim.batchjob.reconType = 'sub'  # 'Deconvolution' or 'Subtraction' or 'None'
    sim.batchjob.plotbinFits = False  # Plots the binned and the non-binned lines, and their fits, during stats only
    sim.batchjob.plotheight = 1
    sim.batchjob.histMax = 50

    # # # # # # # # # # # # # # # # # # # # # # # # # #

    # Misc Flags

    sim.environment.shrinkEnv = True  # Reduce the bitsize of the los data
    sim.batchjob.keepAll = False # This keeps all simulation data, or only the current one
    sim.batchjob.saveSims = False
    printSim = False  # This makes it show the generating profile progress bar

    ##Plotting Flags
    sim.simulate.plotSimProfs = False  # Shows all the little gaussians added up

    sim.batchjob.plotFits = False  # Plots the Different fits to the line w/ the raw line
    sim.batchjob.maxFitPlot = 10

    sim.batchjob.hahnPlot = False  # Plot the green Hahn Data on the primary plot
    sim.batchjob.plotRatio = False  # Plot the ratio of the reconstruction/raw fits

    # Examine Batch Line Profiles
    showProfiles = False  # Plot some number of line profiles at each impact parameter
    maxPlotLines = 30
    average = False
    norm = True
    log = False

    ################# Misc Flags ###############

    # Time Stuff
    timeAx = [0]  # np.arange(0, 30, 3)
    sim.simulate.randTime = True  # adds a random offset to the timeax of each simulate




    ##################################################################################
    # This is where all of the mechanistic call code for the program lives
    ##################################################################################

    # This header handles calling the program in parallel
    try:
        go = int(sys.argv[1])
    except:
        go = 1
    if useMPI and go == 1 and (compute or refineBmin or processEnvironments):
        print("\nStarting MPI...", end='', flush=True)
        os.system("mpiexec -n {} python main.py 0".format(cores))
        print("Parallel Job Complete")
    else:

        ### Process Envs ###
        ####################

        if processEnvironments:
            env = sim.envrs(envsName).processEnv()
            sys.stdout.flush()
        comm.barrier()

        ### Level 3 ### BatchSim
        ########################

        if compute:
            #Load the environment
            env = sim.envrs(envsName).loadEnv()
            env.loadParams(params)
            if params._firstRun:
                if confirm:
                    plt.semilogy(np.ones_like(params.impacts()), params.impacts() - 1, 'o')
                    plt.axhline(1)
                    plt.title("Close Window to Confirm Your Height Selections")
                    plt.ylabel('Solar Radii')
                    plt.show(True)

                # Run the simulation
                myBatch = sim.impactsim(params)
            else:
                # Resume the Simulation
                myBatch = sim.batch(params).restartBatch(env)
        if root:
            if analyze:
                # Get the Environment
                env = sim.envrs(envsName).loadEnv()
                # Analyze the simulation
                myBatch = sim.batch(params).analyzeBatch(env, storeF)


    #### Single Sim Playground ##############

    if simOne and root:
        print('Beginning...')
        df = grid.defGrid()
        env = sim.envrs(envsName).loadEnv(params)
        an = sim.analysis(env)

        # env.fLoad('Remastered2')
        # env.fLoad_lin('Remastered_lin2')
        # env.save()
        # env.fPlot()
        # env.loadAllBfiles()
        # env.create_vec_interps()
        #
        # env.save()
        # env.assignColors()
        # env.save()

        ### PAPER PLOTS ###
        # env.zephyrPlot(True)
        # env.plotElements(True)  # This plots all the charge states of each element
        # env.plotChargeStates(True) # This plots the density of ions of interest
        # env.plotSuperRadial(True)  #This shows the super radial plot
        # an.thermalTempPlot('Wind 0', True)
        # an.losBehavior(0.02, save=True)
        # an.fadeInWindPlotVelocity(save=True)
        # an.windCvRPlotAll(True)
        an.incidentChoices(False)
        # an.reDistContour(False)
        # an.boostLook(save=True)
        # env.makeTable()
        # an.projectionPlot(batchName, True)




        # an.contributionAtHeights()

        # an.losBehavior(0.1)
        # an.losBehavior(1.0)
        # an.losBehavior(5.0)

        # magneticWind()

        # env._LOS2DLoad()
        # env.save()

        # xx = np.linspace(1,100)
        # yy = xx**3
        # plt.plot(xx, yy)
        # plt.show()
        # env.zephyrPlot()

        # for dd in np.arange(10):
        #     env.elements['n'].plotGridRaw(dd)
        # plt.show()
        # env.findFreezeAll()
        # env.plotElements()  # This plots all the charge states of each element
        # env.plotChargeStates() # This plots the density of ions of interest
        # env.plotTotals(True) # This plots all the element total densities
        # env.plotSuperRadial()  #This shows the super radial plot


        # for ion in env.ions:
        # env.plot_ionization(env.ions[0])

        # chopPlots()
        # chopAffectVelocity()
        # contributionAtHeights()
        # fig,ax=plt.subplots()
        # plotContribution(2.25, ax, 50)
        # lineRatioPlot()
        # checkLightTypes()
        # chopAffectTemperature()

        # path = "C:/Users/chgi7364/Dropbox/All School/CU/Steve Research/Weekly Meetings/2019/Meeting 5-14/Velocities"
        # plt.savefig(os.path.abspath('{}/{}.png'.format(path, batchName)))

        # batch1.moranPlot()
        # batch1.reassignColors()
        # batch1.moranFitting()
        # batch1.moranFitPlot()
        # sim.batch('Thermal').renameBatch('Wind_0.00')




        # fadeInWindPlotTemp()
        # saveAllCvR()

        # windCvRPlotAll(env)

        # expectationPlot()
        # expectationPlotScatter()
        # env.params.doChop= False
        # plotContribution()


        if False:
            batchName = 'Wind_1.00'
            batch1 = sim.batch(batchName).loadBatch(env)
            batch1.redoStats = True
            batch1.plotbinFits = False
            batch1.plotFits = False
            batch1.doLinePlot = True
            batch1.linePath = "C:/Users/chgi7364/Dropbox/All School/CU/Steve Research/Weekly Meetings/2019/Meeting 5-28/Lines/{}".format(
                batchName)
            batch1.plotheight = 0
            batch1.doStats()





        ##### PAPER FIGURES #####



        if False:
            z = 1.6
            x = 50
            N = 200 # 'auto'
            position, target = [0, 0, 1], [0, 0, 10]
            primeLineVLong = grid.sightline(position, target, coords='cart')

            params = sim.runParameters()
            # params.maxIons(3)
            params.smooth()
            params.resolution(N, False)
            # params.flatSpectrum(True)
            env = sim.envrs(envsName).loadEnv(params)
            t = time.time()
            lineSim1 = sim.simulate(primeLineVLong, env, getProf=True)
            print("Finished in {:0.3}".format(time.time()-t))
            lineSim1.plot('uw', abscissa='z')
            lineSim1.plot('Te', abscissa='z')
            lineSim1.plot('rho', abscissa='z')

            # lineSim1.plot('N', ion=0, abscissa='cPos', absdim=0, yscale='log')
            # lineSim1.simLine.plotProjections()
            # sim.simulate.vectorize = False
            # lineSim2 = sim.simulate(primeLineVLong, env, N=N, getProf=True)
            # fig, ax = plt.subplots()
            # lineSim1.plotProfiles()
            # lineSim2.plotProfiles(ax, ls='--')

            # toPlot = 'intR'
            # lineSim1.plot(toPlot, ion=0, block=True, yscale='log')#, useax=ax, abscissa='nuAxPrime')
            # lineSim2.plot(toPlot, ion=0, yscale='log', useax=ax, ls='--', single=50, c='r', label=' Serial', abscissa='nuAxPrime')

            # toPlot = 'intR'
            # ion = 0
            # los1 = lineSim1.get(toPlot, ion=ion)
            # abss1 = lineSim1.get('cPos', 0)
            # los2 = lineSim2.get(toPlot, ion=ion)
            # abss2 = lineSim2.get('cPos', 0)
            #
            # plt.plot(abss1, los1, 'b', label='Vector')
            # plt.plot(abss2, los2, 'r--', label='Serial')
            # plt.title(toPlot)
            # plt.yscale('log')
            # plt.legend()
            # plt.show()

            # # The cool new time evolution plots
            # sim.simulate.movName = 'windowPlot.mp4'
            # times = np.linspace(0, 150, 100)
            # # lineSim.evolveLine(times, 10)
            # lineSim.plotProfileT(90, 10)






        if False:
            # Plot the RMS wave amplitudes as fn of height

            # Get the RMS values out of the braid model
            maxImpact = 20
            braidRMS = []
            braidImpacts = np.linspace(1.015, maxImpact, 100)
            for b in braidImpacts:
                braidRMS.append(env.interpVrms(b))

            # Get the RMS values out of coronasim

            rez = 2000
            line = sim.simulate(grid.sightline([1, 0, 0], [maxImpact, 0, 0], coords='Sphere'), env, rez, timeAx=[0])
            modelRMS = line.get('vRms')
            modelImpacts = line.get('pPos', 0)

            plt.plot(braidImpacts, [env.cm2km(x) for x in braidRMS], 'b', label='BRAID')
            plt.plot(modelImpacts, [env.cm2km(x) for x in modelRMS], 'r:', label='GHOSTS')
            plt.legend()
            plt.xlabel('r / $R_{\odot}$')
            plt.ylabel('RMS Amplitude (km/s)')
            plt.title('RMS Wave Amplitude')
            plt.xscale('log')
            plt.show()

        if False:
            # Plot the alfven profile as a fn of height
            maxImpact = 10
            rez = 1000
            line = sim.simulate(grid.sightline([1, 0.5, 0], [maxImpact, 0.5, 0], coords='Sphere'), env, rez,
                                timeAx=[0])
            values = line.get('alfU1')
            abss = line.get('pPos', 0)

            plt.plot(abss, values, 'o-')
            plt.axvline(3)
            plt.title("A wave")
            plt.xlabel("Z")
            plt.ylabel("Wave amplitude")
            plt.show()

        # for ion in env.ions:
        #    print(f"{ion['ionString']}_{ion['ion']} lam00 = {ion['lam00']}: E1 = {ion['E1']}")

        # 1/0
        # env.printFreezeHeights()

        # env.fPlot()
        # env.plot2DV()

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

        # noinspection PyUnreachableCode
        if False:
            # Plot a sightline
            y = 0.001
            x = 20
            z = 1.01
            N = 200

            thisLine = df.poleLine
            thisLine.setEnvInd(0)

            position, target = [x, y, z], [-x, y, z]
            myLine = grid.sightline(position, target, coords='cart')
            lineSim = sim.simulate(thisLine, env, N=N, findT=True, getProf=False, printOut=True)
            # plt.xlim((0.000012382, 0.000012396))
            # plt.yscale('log')
            # plt.show()
            # profR = lineSim.profilesR[0]
            # lamax = np.squeeze(lineSim.ions[0]['lamAx'])
            # plt.plot(lamax, profR, 'k', lw=3)
            ##plt.yscale('log')
            # plt.show()

            ions = [4, 5, 6, 7]

            ax = lineSim.plot(['N', 'Neq'], xscale='log', yscale='log', frame=False, show=True, abscissa='pPos',
                              ion=ions, savename=z)

            # r = 1
            # for ii in ions:
            #    int = np.max(lineSim.ions[ii]['I0array'])
            #    plt.plot(r, int, 'o', label = "SUMER Amplitude")

            # plt.show()
            # lineSim.plot(['radialV', 'ur'], ion = -1, abscissa = 'cPos', frame = False)
        # lineSim.plot('N', ion = -1, abscissa = 'cPos', yscale = 'log', norm = True)
        # lineSim.plot('delta', abscissa='cPos')
        # print(env.interp_f1(3.5))
        # lineSim.plot('uw', abscissa = 'pPos')

        # vlos = lineSim.get('vLOS')
        # plt.hist(vlos, 200)
        # plt.xlabel('Velocity')
        # plt.ylabel('Counts')
        # plt.title('Velocity Distribution along LOS')
        # plt.show()

        # twave = lineSim.get('twave')
        # r = lineSim.get('pPos', dim=0)

        # ans = env.interp(r, twave, 3)
        # print(ans)

        # sim.plotpB()

        # ex = 4

        # xx = np.linspace(0,1,60)
        # kk = xx**ex

        # kmin = 1.0001
        # kmax = 6
        # dif = kmax - kmin
        # qq = kk * dif + kmin

        # env.plot2DV()
        # if self.root:
        #    import pdb
        #    pdb.set_trace()
        # self.comm.barrier()

        # env.plot('ur_raw', 'rx_raw')

        ##For plotting the buildline plots:
        # y = 0.001
        # x = 20
        # z = 2
        # N = 302

        # position, target = [x, y, z], [-x, y, z]
        # myLine = grid.sightline(position, target, coords = 'cart')
        # lineSim = sim.simulate(myLine, env, N = N, findT = True, getProf = True)

        ##This bit here is to plot something at many impacts on the same plot
        # y = 0.001
        # x = 20
        # z = 2
        # N = 302
        # zlist = np.linspace(1.01,5,6).tolist()

        # p1 = 'urProjRel'
        # p2 = 'rmsProjRel'

        # z = zlist.pop(0)
        # position, target = [x, y, z], [-x, y, z]
        # myLine = grid.sightline(position, target, coords = 'cart')
        # lineSim = sim.simulate(myLine, env, N = N, findT = True, getProf = True)
        # ax1 = lineSim.plot(p1, refresh = True, frame = False, show = False, label = z, abscissa = 'cPos', axes = 1e-9)
        # ax2 = lineSim.plot(p2, refresh = True, frame = False, show = False, label = z, abscissa = 'cPos', axes = 1e-9)

        # for z in zlist:
        #    color = next(ax1._get_lines.prop_cycler)['color']
        #    position, target = [x, y, z], [-x, y, z]
        #    myLine = grid.sightline(position, target, coords = 'cart')
        #    lineSim = sim.simulate(myLine, env, N = N, findT = True, getProf = True)
        #    lineSim.plot(p1, refresh = True, useax = ax1, show = False, label = z, c = color, abscissa = 'cPos', axes = False)
        #    lineSim.plot(p2, refresh = True, useax = ax2, show = False, label = z, c = color, abscissa = 'cPos', axes = False)
        # ax1.legend()
        # ax2.legend()
        # plt.show()

        ##lineSim.plotProfile()
        # lineSim.plot('totalInt', ion = -1, frame = False, abscissa='pPos', yscale = 'log')
        # ax = lineSim.plot('frac', ion = -1, yscale = 'log', frame = False, abscissa = 'pPos', ylim = [1e-5,1], xlim = [1,4],show = False)

        # if sim.environment.ionFreeze and True:
        #    ax.set_color_cycle(None)
        #    for ion in env.ions:
        #        color = next(ax._get_lines.prop_cycler)['color']
        #        R = ion['r_freeze']
        #        T = env.interp_T(R)
        #        frac = env.interp_frac(T, ion)
        #        ax.plot(R, frac, color = color, marker = 'o')
        #        print("{}_{}: {}".format(ion['ionString'],ion['ion'], ion['r_freeze']))

        ##lineSim.plot2('dangle', 'pPos', dim2 = 1)
        # lineSim.plot('dPB', linestyle = 'o', scaling = 'log')

        # env.fPlot()
        if False:
            # Plot a plane
            lineSim = sim.simulate(df.polePlane, env, N=400, findT=False, getProf=False, printOut=True)
            lineSim.plot('alfU1', cmap='RdBu', threeD=False, sun=True)

            # lineSim.plot('streamIndex', sun = True) #, ion = -1, abscissa = 'T', yscale = 'log')
        # lineSim.plot('densfac')

        ###ax = lineSim.plot('rho', scaling = 'log', frame = False, vmax = -16, vmin = -21, cmap = 'inferno', clabel = 'log($g/cm^2$)', suptitle = 'Mass Density', extend = 'both')
        # ax2 = lineSim.plot('alfU1', cmap = 'RdBu', center = True, clabel='km/s', show = False)
        # streams = lineSim.get('streamIndex')
        # env.plotEdges(ax2, streams, False)
        # plt.show()

        # ax2 = lineSim.plot('pU', dim = 1, cmap = 'RdBu', center = True, clabel='km/s')

        # temps, _ = lineSim.get('T')
        # rad, _ = lineSim.get('pPos', dim = 0)
        # F = []
        ##temps = np.logspace(4,8,500)
        # for T in temps:
        #    #print(T)
        #    F.append(env.interp_frac(T))
        ##plt.plot(temps)
        # plt.plot(rad, F, label = 'Frac')
        ##plt.plot(rad, np.log10(temps)/18, label = 'Temp')

        # plt.yscale('log')
        ##plt.plot(10**env.chTemps,env.chFracs, label = 'Raw')
        # plt.legend()
        # plt.show()
        # lineSim = sim.simulate(df.primeLineVLong, env, N=2000, findT=False, getProf=False, printOut=True)
        #
        # # The cool new time evolution plots
        # T = 800
        # sim.simulate.movName = 'windowPlot.mp4'
        # lineSim.evolveLine(T,0,T)

        # lineSim.plot('T', scaling = 'log', abscissa='pPos')
        # lineSim.plot('frac', abscissa = 'pPos')
        # lineSim.plot('alfU2', cmap = 'RdBu', center = True, abscissa = 'pPos', absdim = 0)
        # lineSim.compare('uTheta', 'pU', p2Dim = 1, center = True)
        # lineSim.quiverPlot()

        ##lineSim.plot2('frac','nion', p1Scaling='log', p2Scaling='log')
        # lineSim.plotProfile()

        # bpoleSim = sim.simulate(df.bpolePlane, env, N = 200, findT = False, printOut = True)
        # bpoleSim.getProfile()
        # bpoleSim.plot('totalInt', cmap = 'brg', scaling = 'root', scale = 15)

    # position, target = [10, 3, 1.5], [-10, -3, 1.5]
    # cyline = grid.sightline(position, target, coords = 'Cart', rez = None, size = [0.002,0.01])

    # timeAx = [0] #np.arange(0,1500)
    # cylSim = sim.simulate(cyline, env, [1500, 3000], 1, False, True, timeAx = timeAx)
    # cylSim.getProfile()

    # cylSim.plot('densfac')
    # cylSim.plot2('vLOS','vLOSwind')
    # myBatch = sim.impactsim(*params, pBname = pBname)
    # if rank == 0:
    #    pBgoal = np.asarray(myBatch.pBavg, dtype = 'float')
    # else:
    #    pBgoal = np.empty(1, dtype = 'float')
    # comm.Bcast(pBgoal, root=0)

    # df = grid.defGrid()

    # env = sim.envs(envsName).loadEnvs(1)[0]

    # bpoleSim = sim.simulate(df.bpolePlane, env, N = 500, findT = False, printOut = True)
    # bpoleSim.setTime()
    # bpoleSim.plot('alfU1', cmap = 'RdBu', center = True)

    # bpoleSim.plot('vLOS', scaling = 'none', cmap = 'RdBu' )

    # lineSim2 = sim.simulate(df.primeLine, env, N = (1000,10000), findT = True)
    # print((lineSim2.Npoints))
    # lineSim2.evolveLine(200, 0, 200)

    ## Misc Sims ##
    # topSim = sim.simulate(df.topPlane, N = 1000)
    # poleSim = sim.simulate(df.polePlane, findT = True, step = 0.01)

    ## bpoleSim ##
    # t = time.time()

    # t = time.time()
    # print('Elapsed Time: ' + str(time.time() - t))
    # bpoleSim.compare('rho', 'intensity', p1Scaling = 'log', p2Scaling = 'log')

    # print('Go')
    # lineSim = sim.simulate(df.impLine, env, N = 1500, findT = True)
    # lineSim.peekLamTime()

    # poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)
    # lineSim.plotProfile()
    # print(lineSim.getStats())
    # poleLineSim = sim.simulate(df.poleLine, findT = True, N = 1000)

    # env = sim.envs('smoothEnvs').loadEnvs(1)[0]

    # lineSim2.setIntLam(200.01)
    # lineSim2.plot('intensity')
    # print(lineSim2.getStats())
    ## Useful Functions
    # mysim.plot('property')
    # mysim.Keys()
    # mysim.timeV(t0,t1,step)

    # topSim.plot('uPhi')
    # bpoleSim.plot('uPhi')
    # poleLineSim.plot('vGrad')
    # lineSim.evolveLine()
    # lineSim.timeV(t1 = 2000, step = 3)
    # bpoleSim.plot('T')
    # sim.simpoint()

    # The whole code in one line. WOO!
    # mySim = sim.simulate(sim.defGrid().bpolePlane, step = 0.1)#.plot('rho', scale = 'log')

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()
    # root = rank == 0

    ### Level 2 ### MultiSim
    ###############

    # lines = grid.rotLines(6)
    # lineSims = sim.multisim(lines, env, N = 200)
    # lineSims.plotLines()
    # plt.imshow(np.log(lineSims.getLineArray()))
    # plt.show()

    # lines = grid.rotLines()
    # lineSims = sim.multisim(lines, env, N = 1000)
    # plt.pcolormesh(np.log(lineSims.getLineArray()))
    # plt.show()

    ### Level 0 ### Simpoint
    ###############
    # env = sim.envs(envsName).loadEnvs(1)[0]
    # df = grid.defGrid()
    # thisPoint = sim.simpoint(grid = df.bpolePlane, env = env)
    # thisPoint.setTime()
    # thisPoint.show()

    # if root:
    #    print('')
    #    print('Sim Name =')
    #    print([x for x in vars().keys() if "Sim" in x])

    # if root:
    #    print('')
    #    print('Batch Name =')
    #    print(batchName)

    if root: print("\nEnd of Program")

