import masterslave as ms
import numpy as np
import progressBar as pb
from mpi4py import MPI
import matplotlib.pyplot as plt
comm = MPI.COMM_WORLD
root = comm.rank == 0

def mainFunc(N):
    return N**3

# def receivFunc(data):
    # global bar
    # bar.increment()
    # bar.display()
    
x = np.linspace(0,4*3.14159265358,100000).tolist()
# if root:
    # bar = pb.ProgressBar(len(x))
    # bar.display()
    
# ms.set_response(receivFunc)
ans, ind = ms.poolMPI(x, mainFunc, True, True)




# if root: 
new = ms.reorderScalar(ans, ind)
plt.plot(x, ans)
plt.plot(x, new)
plt.show()
        
print('\nFinished')
    # print("list:{}, finish:{}".format(len(a), len(ans)))