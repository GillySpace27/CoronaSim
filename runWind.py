import coronasim as sim
import numpy as np
import sys

# Simulation Parameters
baseName = 'Wind {}'
sim.simpoint.useB = False
sim.simpoint.g_useWaves = False
sim.simpoint.g_useWind = True
impactPoints = 100
b0 = 1.01
b1 = 11
spacing = 'log'
N_line = 'auto'
iterations = 1
firstRun = True

# Distribute the Parameters
envsName = 'Remastered'
names = ['0 FullChop', '0 Chop', '100 FullChop', '100 Chop', '100', '75', '50', '25', '0']
speeds = ['0.00', '0.00', '1.00', '1.00', '1.00', '0.75', '0.50', '0.25', '0.00']
chop = [True, True, True, True, False, False, False, False, False]
keepPump = [False, True, False, True, False, False, False, False, False]

try:
    rank = int(sys.argv[1])
except:
    sys.exit("No Initial Conditions Provided")

sim.simpoint.windFactor = float(speeds[rank])
sim.simpoint.doChop = chop[rank]
sim.simpoint.keepPump = keepPump[rank]
batchName = baseName.format(names[rank])

# Create the impact array
if b1 is not None:
    if spacing.casefold() in 'log'.casefold():
        logsteps = np.logspace(np.log10(b0 - 1), np.log10(b1 - 1), impactPoints)
        impacts = np.round(logsteps + 1, 5)
    else:
        impacts = np.round(np.linspace(b0, b1, impactPoints), 4)
else:
    impacts = np.round([b0], 4)

# Load the environment
env = sim.envrs(envsName).loadEnv()

if firstRun:
    # Run the simulation
    myBatch = sim.impactsim(batchName, env, impacts, iterations, N_line)#, printQuiet=True, allSave=True)
else:
    # Resume the Simulation
    myBatch = sim.batch(batchName).restartBatch(env)


