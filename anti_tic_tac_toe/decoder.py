import sys
import numpy as np
from numpy.core.defchararray import index


val_file = sys.argv[2]
sfile = sys.argv[4]
pid = sys.argv[6]

astates = []
pol = []

file = open(sfile,"r")
lines = file.readlines()
for line in lines:
    line = line.split()
    astates.append(line[0])
file.close()

file = open(val_file,"r")
lines = file.readlines()
for line in lines:
    line = line.split()
    pol.append(int(float(line[1])))
file.close()

if __name__ == "__main__":
    print(pid)
    for i in range(len(astates)):
        act = np.zeros(9)
        act[pol[i]] = 1
        print(astates[i], act[0], act[1], act[2], act[3], act[4], act[5], act[6], act[7], act[8])