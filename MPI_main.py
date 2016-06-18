
import numpy as np
#import progressBar as pb
import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim
import time


df = grid.defGrid()
lines = df.impactLines(50)


linestats = sim.coronasim_MPI(lines, N = 1000, findT = True)
