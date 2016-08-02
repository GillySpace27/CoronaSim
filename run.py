import os
import sys
argv = sys.argv
name = argv.pop(0)
if len(argv) > 0:
    num = argv.pop(0)
else: num = 7

    
os.system("mpiexec -n " + str(num) +" python main.py")
