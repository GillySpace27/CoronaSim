from mpi4py import MPI
comm = MPI.COMM_WORLD

print("I ran on core {}".format(comm.Get_rank()))