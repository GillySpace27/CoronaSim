#Level 1

          # if not self.adapt:
            # self.cPoints = self.grid.cGrid(self.N, iL = self.iL)

            # if type(self.grid) is grid.plane:
                # self.adapt = False

            # self.Npoints = len(self.cPoints)
            # self.shape = self.grid.shape
            # self.shape2 = [*self.shape, -1]              
                
                
        # if adaptable:               
        # else:
            #Rigid Mesh
            # t = time.time()
            # nnnn = 0
            # step = 1/self.N
            # for cPos in self.grid: 
                # nnnn += 1
                # thisPoint = simpoint(cPos, self.grid, self.env, self.findT) 
                # self.sPoints.append(thisPoint)
                # self.steps.append(step)
                # self.pData.append(thisPoint.Vars())
                # if doBar: 
                    # bar.increment()
                    # bar.display()
            # if doBar and self.print: bar.display(force = True)
            # if self.print: print('Elapsed Time: ' + str(time.time() - t))


            
            ##Parallel Way
            #chunkSize = 1e5
            #print('  Initializing Pool...')
            #pool = Pool()
            #thisPoint = simpoint(grid = self.grid)
            #for pnt in pool.imap(partial(simpoint, grid = self.grid, env = self.env, findT = self.findT), self.cPoints, chunksize):
            #    self.sPoints.append(pnt)
            #    self.pData.append(pnt.Vars())
            #    bar.increment()
            #    bar.display()
            #bar.display(force = True)
            #pool.close()
            #pool.join()

        #print('')
        #print('')


    #def simulate_MPI(self):
    #    #Break up the simulation into multiple PEs
    #    comm = MPI.COMM_WORLD
    #    rank = comm.Get_rank()
    #    size = comm.Get_size()
    #    assert len(self.cPoints) % size == 0
    #    chunksize = len(self.cPoints) / size
    #    if rank == 0: t = time.time()
    #    local_data = []
    #    local_simpoints = []
    #    local_pdata = []
    #    #if rank ==0:
    #    #    data = self.cPoints
    #    #else:
    #    #    data = None
    #    #local_data = comm.scatter(data, root = 0)
    #    #if rank == 0:
    #    low = int(rank*chunksize)
    #    high = int(((rank+1)*chunksize))
    #    local_data = self.cPoints[low:high]
    #    #else: local_data = self.cPoints[int(rank*chunksize+1):int(((rank+1)*chunksize))]
    #    for cPos in local_data:
    #        thisPoint = simpoint(cPos, self.grid, self.env, self.findT) 
    #        local_simpoints.append(thisPoint)
    #        local_pdata.append(thisPoint.Vars())

    #    sListList = comm.gather(local_simpoints, root=0)
    #    pdatList = comm.gather(local_pdata, root=0)
    #    if rank ==0:
    #        self.sPoints = []
    #        self.pData = []
    #        for list in sListList:
    #            self.sPoints.extend(list)
    #        for pdat in pdatList:
    #            self.pData.extend(pdat)
    #        print("length = ")
    #        print(len(self.pData))
    #        print('GO')
    #        print('Elapsed Time: ' + str(time.time() - t))

    #    else: 
    #        print("Proccess " + str(rank) + " Complete")
    #        sys.exit(0)



#Rand and Stream Velocities

    #def findRandU(self):
    #    #Wrong
    #    #amp = 0.01 * self.vRms * (self.rho / self.rho_min)**1.2
    #    #rand = simpoint.thermRand.standard_normal(3)
    #    return 0,0,0 #amp*rand

    #def findStreamU(self):
    #    #if self.streamIndex == 0: return 0.0, 0.0 #Velocities for the regions between streams
    #    #simpoint.streamRand.seed(int(self.streamIndex))
    #    #thisRand = simpoint.streamRand.standard_normal(2)
    #    ##Wrong scaling
    #    #streamTheta = 0.1*self.vRms*thisRand[0]
    #    #streamPhi = 0.1*self.vRms*thisRand[1]
    #    return 0,0 #streamTheta, streamPhi


            #self.streamTheta, self.streamPhi = self.findStreamU() #All Zeros for now
            #self.randUr, self.randTheta, self.randPhi = self.findRandU() #All Zeros for now
            #self.uTheta = self.streamTheta + self.randTheta + 0.05* self.vRms*self.xi(t - self.twave)
            #self.uPhi = self.streamPhi + self.randPhi + 0.05*self.vRms*self.xi(t - self.twave) 


#Density stuff

        #self.num_den_min = self.minNumDense(self.rx)
        #self.num_den = self.densfac * self.num_den_min
        #self.rho_min = self.num2rho(self.num_den_min)
        #self.rho = self.num2rho(self.num_den)
        
    #def minNumDense(self, rx):
    #    #Find the number density floor
    #    return self.interp_rx_dat(self.num_dens_raw)
    #    #return 6.0e4 * (5796./rx**33.9 + 1500./rx**16. + 300./rx**8. + 25./rx**4. + 1./rx**2)
        
    #def num2rho(self, num_den):
    #    #Convert number density to physical units (cgs)
    #    return (1.841e-24) * num_den  






    #def gauss_function(x, a, x0, sigma):
    #    return a*np.exp(-(x-x0)**2/(2*sigma**2))





    #def plotStats(self):
    #    f, axArray = plt.subplots(5, 1, sharex=True)
    #    mm = 0
    #    titles = ['amp', 'mean', 'sigma', 'skew', 'kurtosis']
    #    ylabels = ['', 'Angstroms', 'Angstroms', '', '']
    #    for ax in axArray:
    #        if mm == 0:
    #            ax.plot(self.gridLabels, np.log([x[mm] for x in self.lineStats]))
    #        else:
    #            ax.plot(self.gridLabels, [x[mm] for x in self.lineStats])
    #        ax.set_title(titles[mm])
    #        ax.set_ylabel(ylabels[mm])
    #        mm += 1
    #    ax.set_xlabel('Impact Parameter')
    #    plt.show()




    ##def timeV(self, t0 = 0, t1 = 100, tstep = 2):
    ##    print('Timestepping...')
    ##    self.times = np.arange(t0, t1, tstep)
    ##    bar = pb.ProgressBar(self.Npoints*len(self.times))
    ##    stdList = []
    ##    vMeanList = []
    ##    for tt in self.times:
    ##        thisV = []
    ##        for point in self.sPoints:
    ##            point.setTime(tt)
    ##            thisV.append(point.gradV(self.grid.ngrad))
    ##            bar.increment()
    ##            bar.display()
    ##        stdList.append(np.std(np.array(thisV)))
    ##        vMeanList.append(np.mean(np.array(thisV)))           
    ##    bar.display(force = True)
    ##    self.vStd = np.array(stdList)
    ##    self.vStdErr = self.vStd / np.sqrt(len(stdList))
    ##    self.vMean = np.array(vMeanList)
    ##    self.timePlot()

    ##def timePlot(self):
    ##    #plt.semilogy(self.times, np.abs(self.vStd), self.times, np.abs(self.vSum))
    ##    plt.figure()
    ##    plt.plot(self.times, self.vMean)
    ##    plt.plot(self.times, self.vStdErr)
    ##    plt.title('mean and std of gradV over time')
    ##    plt.show(block = False)



    #def getPointSpeeds(self, pPos):
    #    #Find all of the velocities for an arbitrary point
    #    u = self.findU(pPos)
    #    vAlf = self.findAlf(pPos)
    #    vPh = self.u + self.vAlf
    #    return vPh, u, vAlf
    
    #def getPointDensity(self, pPos):
    #    #Find the density of another grid point
    #    rx = self.r2rx(pPos[0])
    #    B = self.findB(pPos, self.BMap)
    #    num_den_min = self.minNumDense(rx)
    #    num_den = self.actualDensity(num_den_min, B)
    #    return self.num2rho(num_den), num_den