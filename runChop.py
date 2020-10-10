import coronasim as sim

if sim.runParallel():
    params = sim.runParameters()
    env = sim.envrs(params.envName()).loadEnv(params)
    an = sim.analysis(env)
    an.chop_simulation()

