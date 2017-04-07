import os
import sys
argv = sys.argv
name = argv.pop(0)
if len(argv) > 0:
    num = argv.pop(0)
else: num = 46

print("Starting run with {} cores.".format(num))
os.system("mpiexec -n {} python3 main.py {}".format(num, 2))
