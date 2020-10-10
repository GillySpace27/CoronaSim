## This is what MAIN was like when I submitted the first paper.
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


    # # # Which part of the program should run? # # # #
    simOne = True
    compute = False
    analyze = False

    force_mpi = False

    # Batch Name
    batchName = 'HQProj_U_100b'#'HQProj025'
    # batchName = 'Boost3 1'
    params = sim.runParameters(batchName)
    params.firstRun(True)  # Overwrite?

    # # # # # # Compute Properties # # # # # # # # # # #

    params.impacts(1.01, 11, 2)

    params.resolution(10)
    # params.xrange(75)
    # params.maxIons([7,10,11])
    # params.lamPrimeRez(200)
    # params.lamRez(250)

    params.useWind(True)
    params.useB(False)
    params.useWaves(False)

    params.windFactor(1)
    params.doChop(False) # Cut out the incident continuum
    params.makeLight(True)

    sim.batchjob.redoStats = False


    # Run in parallel?
    sim.batchjob.usePool = False
    useMPI = False
    cores = 5

    confirm = False
    params.compute = compute
    params.analyze = analyze

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
    try:
        # This header handles calling the program in parallel
        go_for_mpi = useMPI and size == 1 and (compute or refineBmin or processEnvironments or force_mpi)
        if go_for_mpi:
            sim.runParallel()
        else:

            ### Process Envs ###
            ####################

            if processEnvironments:
                env = sim.envrs(envsName).processEnv()
                sys.stdout.flush()
            comm.barrier()

            ### Compute ########
            #####################

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
                    myBatch = sim.batch(params).restartBatch()
            if root:
                if analyze:
                    # Get the Environment
                    env = sim.envrs(envsName).loadEnv()
                    # Analyze the simulation
                    params.compute=False
                    myBatch = sim.batch(params)
                    myBatch.analyzeBatch(storeF)

            plt.show()
    except Exception as e:
        raise e


    #### Single Sim Playground ##############

    if simOne and root:
        print('Beginning...')
        df = grid.defGrid()
        env = sim.envrs(envsName).loadEnv(params)
        an = sim.analysis(env)
        # A = env.interp_T(1.02)
        # B = env.interp_T(2)
        # print(A)
        # print(B)
        # print(B/A)
        # print((B-A)/A)

        # rx = np.linspace(1.0, 1000, 10)
        # SA = env.findSunSolidAngle(rx)
        # print(SA)
        # plt.plot(rx, SA)
        # plt.show()


        # bat = sim.batch('Boost4 1')
        # import pdb; pdb.set_trace()
        # bat.getProfiles(True)

        # bat = sim.batch('Wind 100 FullChop')
        # bat.rename('Wind 100 FullChop Both')
        # del bat
        # env.measure(bat)
        # env.measure(env)
        #
        # env.fLoad('current')
        # env.fLoad_lin('Remastered_lin2')
        # env.save()
        # env.fPlot()
        # env.loadAllBfiles()
        # env.create_vec_interps()
        #
        # env.save()
        # env.assignColors()
        # env.save()

        # PAPER 1 PLOTS ###
        env.zephyrPlot(False)
        # env.plotSuperRadial(True)  #This shows the super radial plot
        # env.plotElements(True)  # This plots all the charge states of each element
        # env.plotChargeStates(True) # IL# This plots the density of ions of interest
        # an.thermalTempPlot(save=False, batchName="Boost3 1") #IL#
        # an.losBehavior(0.015, save=True) #IL# #Compute#
        # an.validateReduction(False, bat1Name=batchName) #IL#
        # an.validateReduction_thermal(False) #IL#
        # an.fadeInWindPlotVelocity(save=True) #IL#
        # an.figProj_wind(True)
        # an.figProj_wind_boost_square('both', "Boost5")
        # an.moranPlot(True)
        # an.incidentChoices_square(True) #Compute#
        # an.incidentChoices_square_nopump('both') #Compute#
        # an.chop_simulation()
        # an.chopRatio('both')

        # fig, ax = plt.subplots(1,1)
        # ax.set_title("Difference with and without continuum")
        # ax.set_ylabel("Ratio")
        # ax.set_xlabel("Observation Height")
        # ax.set_xscale('log')
        # ax.axhline(1, ls="--")
        # for ii in [0,2]:
        #     # env.doPlotIncidentArray(ii)
        #
        # an.incidentChoices_square_nopump(True, ii) #Compute#
        # plt.show()

        # an.reDistContour(True) # LONG COMPUTE #
        # for ii in np.arange(12):
        #     an.boostLook(save=True, batchName='Boost3', ionNum=ii)
        # env.makeTable2()
        # an.windCvRPlotAll_square_all(True) #IL#
        # an.plotIntensity(False) #IL#


        # env.zephyrPlot_powerpoint('both')
        # env.zephyrPlot_poster('both')
        # env.plotChargeStates_poster('both') # IL# This plots the density of ions of interest
        # an.thermalTempPlot_poster(save=True) #IL#
        # an.validateReduction_poster('both')

        # an.projectionPlot(saveFig=True) #IL#
        #
        # if True:
        #     z = 1.6
        #     x = 50
        #     N = 200  # 'auto'
        #     position, target = [-1000, 0, 8], [1000, 0, 8]
        #     primeLineVLong = grid.sightline(position, target, coords='cart')
        #
        #     params = sim.runParameters()
        #     # params.maxIons([10])
        #     params.smooth_steady()
        #     params.resolution(100, False)
        #     # params.lamPrimeRez(2000)
        #     # params.lamRez(300)
        #     # params.flatSpectrum(True)
        #     env = sim.envrs(envsName).loadEnv(params)
        #     t = time.time()
        #     lineSim1 = sim.simulate(primeLineVLong, env, getProf=True)
        #     print("Finished in {:0.3}".format(time.time() - t))
        #     abss = 'x'
        #     # lineSim1.plot('uw', abscissa=abss)
        #     # lineSim1.plot('Te', abscissa=abss)
        #     # lineSim1.plot('rho', abscissa=abss, yscale='log')
        #     lineSim1.plot('intR', abscissa=abss, yscale='log', ion=-1, wrap=True)





    if root: print("\nEnd of Program")

