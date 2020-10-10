import coronasim as sim

params = sim.runParameters()
batchName = "Boost6"
params.smooth_steady()
# params.useWind(False)
# params.firstRun(False)
# params.makeLight(False)

boosts = [1, 2, 4, 8, 16, 32, 64, 128]

if True:
    # Quick Run
    # boosts = [1, 5, 10, 50]
    params.impacts(bn=4)
    params.resolution(250)

for bb in boosts:
    params.batchName("{} {}".format(batchName, bb))
    params.T_boost(bb)
    sim.batch(params)

# No Wind Case
bb = 1
params.batchName("{} {} noWind".format(batchName, bb))
params.T_boost(bb)
sim.batch(params)

# an = sim.analysis()
# an.boostLook(batchName=batchName, save=False)
