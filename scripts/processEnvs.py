
import numpy as np
#import progressBar as pb
import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim
import time
from mpi4py import MPI


if __name__ == '__main__':

    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # size = comm.Get_size()
    # root = rank == 0
    # envPath = '../dat/primeEnv'
    
    
    # if root:
        # print('\nCoronaSim!')
        # print('Written by Chris Gilbert')
        # print('-------------------------\n')
    # Simulate Common Grids
    # df = grid.defGrid()
    #Initialize Simulation Environment
        # env = sim.environment()
        # env.save(envPath)

    # else:
        # df = None
        # envs = None

    # df = comm.bcast(df, root = 0)
    # envs = comm.bcast(envs, root = 0)
    envs = sim.envs('hybridEnv').processEnvs()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
