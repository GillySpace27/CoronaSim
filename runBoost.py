import coronasim as sim

params = sim.runParameters()
params.firstRun(True)
# params.makeLight(True)
params.smooth()
params.impacts(bn=30)
# params.resolution(100)

boosts = [1,2,4,8,16,32,64,128]

for bb in boosts:
    params.batchName("Boost {}".format(bb))
    params.T_boost(bb)
    sim.batch(params)
#
an = sim.analysis()
an.boostLook()
