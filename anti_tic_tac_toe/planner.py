import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import hsplit
import pulp
from pulp import *
import datetime

mdp = sys.argv[2]
if len(sys.argv)>3:
    al = sys.argv[4]
else:
    al = "vi"

file = open(mdp,"r")
lines = file.readlines()
ns = int(lines[0].split()[1])
na = int(lines[1].split()[1])
tmat = np.zeros((ns,ns,na))
rmat = np.zeros((ns,ns,na))
i = 3
while lines[i].split()[0] != "mdptype":
    s1 = int(lines[i].split()[1])
    a = int(lines[i].split()[2])
    s2 = int(lines[i].split()[3])
    r = float(lines[i].split()[4])
    p = float(lines[i].split()[5])
    # print(s1,a,s2,r,p)
    tmat[s1][s2][a] = p
    rmat[s1][s2][a] = r
    i+=1
gama = float(lines[i+1].split()[1])

file.close()

def val_ite():
    vt = np.zeros(ns)
    pist = np.zeros(ns)
    diff = 1
    while(diff>0.00000001):
        vo = np.copy(vt)
        for i in range(ns):
            tr = np.multiply(tmat[i],rmat[i]+gama*vt[:,np.newaxis])
            suma = np.sum(tr, axis=0)
            vt[i] = np.max(suma)
            pist[i] = np.argmax(suma)
        diff = np.max(vt-vo)
    return np.round(vt,6), np.round(pist,6)

def get_v(tmat,rmat,pol):
    tmatp = np.zeros((ns,ns))
    rmatp = np.zeros((ns,ns))
    for i in range(ns):
        for j in range(ns):
            tmatp[i][j] = tmat[i][j][pol[i]]
            rmatp[i][j] = rmat[i][j][pol[i]]
    vt = np.linalg.solve((np.identity(ns)-gama*tmatp),(np.sum(np.multiply(tmatp,rmatp),axis=1)))
    return vt

def how_pol():
    ist = np.ones(ns)
    pol = [np.random.randint(na) for i in range(ns)]
    
    q = np.zeros((ns,na))
    while(np.any(ist)):
        for i in range(ns):
            vt = get_v(tmat,rmat,pol)
            q[i] = np.sum(np.multiply(tmat[i],rmat[i]+gama*vt[:,np.newaxis]), axis=0)
            qmax = np.max(q[i])
            if qmax-vt[i] > 1e-8:
                pol[i] = np.argmax(q[i])
            else:
                ist[i]=0
        vt = np.round(vt,6)
    return vt, pol

def lp():
    prob = pulp.LpProblem("ValueFn")
    dv = []
    for i in range(ns):
        dv.append(pulp.LpVariable("{}".format(i)))
    prob += lpSum(dv)
    for i in range(ns):
        for j in range(na):
            prob += dv[i] >= pulp.lpSum([(tmat[i][sd][j])*(rmat[i][sd][j]+gama*dv[sd]) for sd in range(ns)])
    # prob.writeLP("LinearProgramming.lp")
    optimization_result = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    vt = np.zeros(ns)
    for a in prob.variables():
        id = int(a.name)
        vt[id] = round(a.varValue,6)
    q = np.zeros((ns,na))
    pol = np.zeros(ns)
    for i in range(ns):
        q[i] = np.sum(np.multiply(tmat[i],rmat[i]+gama*vt[:,np.newaxis]), axis=0)
        pol[i] = np.argmax(q[i])
    return vt,pol

        
if __name__ == "__main__":
    if(al=='hpi'):
        v,pi = how_pol()
    if(al=='vi'):
        v,pi = val_ite()
    if(al=='lp'):
        v,pi = lp()
    for i in range(len(v)):
        print("{0:6f}".format(v[i]), pi[i])