
import numpy as np
import matplotlib
matplotlib.use('tkagg')

import matplotlib.pyplot as plt
import gridgen as grid
import coronasim as sim
import progressBar as pb


if True:             
    #Above the Pole        
    iL = 1
    step = 0.001
    normal = [0,0,1] 
    offset = [1.5, 0, 0]

else:        
    #Slice of the Pole
    iL = 2
    step = 0.001
    normal = [1,0,0] 
    offset = [0, 2, -2]
            

print("Beginning Simulation...")

#Find Density
myPlane = grid.plane(normal, r = offset)
fig = myPlane.plot(iL = iL)
    
n = 0
density=[]
planePoints = myPlane.cPlane(step, iL)
N = len(planePoints)

bar = pb.ProgressBar(N)
print('')

for pos in planePoints:
    density.append(sim.simpoint(pos, findT = False).rho)
    bar.increment()
    bar.display()
bar.display(force = True)

                 

dens = np.asarray(density)
denss = dens.reshape(myPlane.shape)
    
#    np.savetxt('density.txt', denss)
ax = fig.add_subplot(122)
ax.imshow(np.log(denss), interpolation='none')
    
plt.tight_layout()
grid.maximizePlot()
plt.show()











    #    sightline().plot() 

#    position, target = [0,-2,0], [0,2,0]
#    
#    myLine = SightLine(position, target)
    

            
#    plt.imShow(density)
    
#    print(myLine.pLine(0.2))

#    sim.simpoint(myLine.cPoint(0.8)).show()        
        
    
    #myLine.look(target, position)
    
#    line = myLine.cLine(0, 1, 5)
#    print(*line)
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111, projection='3d')
    
#    for jj in np.arange(0,1.1,0.1):
#        ax.scatter(*myLine.cLine(jj))
#    #    print(myLine.cLine(jj))
#    
#    plt.show()
    


#    property_names=[p for p in dir(SomeClass) if isinstance(getattr(SomeClass,p),property)]
#    print(property_names)
#    print(vars(myPoint))
#    myPoint.show()
    
