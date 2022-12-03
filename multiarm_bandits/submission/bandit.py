import sys
import numpy as np
import matplotlib.pyplot as plt


insta = sys.argv[2]
al = sys.argv[4]
rs = int(sys.argv[6])
ep = float(sys.argv[8])
c = float(sys.argv[10])
th = float(sys.argv[12])
hz = int(sys.argv[14])

np.random.seed(rs)

def openfile(ins):
    file = open(ins,"r")
    lines = file.readlines()
    inst = []
    for line in lines:
        line = line.rstrip('\n')
        p = float(line)
        inst.append(p)
    file.close()
    instance = np.array(inst)
    return instance

def openfile_t34(ins):
    file = open(ins,"r")
    lines = file.readlines()
    inst = []
    for line in lines:
        elem = []
        line = line.split()
        for i in line:
            elem.append(float(i))
        inst.append(elem)
    file.close()
    instance = np.array(inst)
    return instance

def pull_arm_t34(ins,n):
    sup = ins[0]
    prob = ins[n+1]
    op = np.random.choice(sup,1, p=prob)
    return op

def pull_arm(ins,n):
    p = ins[n]
    op = np.random.choice(np.arange(0,2), p=[1-p,p])
    return op

def epsilong2(ep,hz,ins):
    l = len(ins)
    reward = np.zeros(l)
    num = np.ones(l)
    for k in range(l):
        out = pull_arm(ins,k)
        reward[k]+=out
    t1 = ep*hz
    for t in range(l,hz):
        if(t<=t1):
            n = np.random.choice(np.arange(0,l))
            out = pull_arm(ins,n)
            reward[n]+=out
            num[n]+=1
            avg = reward/num
        else:
            n = avg.argmax()#can add condition for same avg
            out = pull_arm(ins,n)
            reward[n]+=out
            num[n]+=1
            avg = reward/num
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg

def epsilong3(ep,hz,ins):
    l = len(ins)
    reward = np.zeros(l)
    num = np.zeros(l)
    avg = np.zeros(l)
    #wtd = np.random.choice(np.arange(0,2), p=[ep,1-ep])
    for t in range(0,hz):
        wtd = np.random.rand()
        if(wtd<=ep):
            n = np.random.choice(np.arange(0,l))
            out = pull_arm(ins,n)
            reward[n]+=out
            num[n]+=1
            avg[n] = reward[n]/num[n]
        else:
            mavg=np.max(avg)
            nmaxs=[k for k in range(l) if avg[k]==mavg]#tiebreaker
            n=np.random.choice(nmaxs)
            out = pull_arm(ins,n)
            reward[n]+=out
            num[n]+=1
            avg[n] = reward[n]/num[n]
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg

def ucbf(ins,hz):
    l = len(ins)
    num = np.ones(l)
    reward = np.zeros(l)
    avg = np.zeros(l)
    for k in range(l):
        out = pull_arm(ins,k)
        reward[k]+=out
    avg = reward/num
    ucb=np.copy(avg)

    for t in range(l,hz):
        mucb=np.max(ucb)
        nmaxs=[k for k in range(l) if ucb[k]==mucb]#tiebreaker
        n=np.random.choice(nmaxs)
        out = pull_arm(ins,n)
        reward[n]+=out
        num[n]+=1
        avg = reward/num
        ucb = avg + np.sqrt(2*np.log(t)/num)
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg

def kl(em,q):
        if em==0:
            em=0.00001
        return em*np.log(em/q) + (1-em)*np.log((1-em)/(1-q))

def getmax(avg,ct,l):
    vals = np.zeros(l)
    for j in range(l):
        min_val = avg[j]
        max_val = 1
        for i in range(10):
            val = (min_val+max_val)/2
            if val==min_val:
                break
            if kl(avg[j],val)<=ct[j]:
                min_val = val
            else:
                max_val = val
        vals[j] = min_val
    return vals

def klucbf(ins,hz):
    l = len(ins)
    num = np.ones(l)
    reward = np.zeros(l)
    avg = np.zeros(l)
    for k in range(l):
        out = pull_arm(ins,k)
        reward[k]+=out
    avg = reward/num
    for t in range(l,hz):
        ct = (np.log(t+1) + 3*np.log(np.log(t+1)))/num
        klucb = getmax(avg,ct,l)
        mklucb=np.max(klucb)
        nmaxs=[k for k in range(l) if klucb[k]==mklucb]#tiebreaker
        n=np.random.choice(nmaxs)
        out = pull_arm(ins,n)
        reward[n]+=out
        num[n]+=1
        avg[n] = reward[n]/num[n]
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg

def thompson(ins,hz):
    l = len(ins)
    num = np.ones(l)
    reward = np.zeros(l)
    succ = np.zeros(l)
    fail = np.zeros(l)
    pval = np.random.beta(succ+1,fail+1)
    for t in range(0,hz):
        mpv = np.max(pval)
        nmaxs=[k for k in range(l) if pval[k]==mpv]#tiebreaker
        n=np.random.choice(nmaxs)
        out = pull_arm(ins,n)
        reward[n]+=out
        num[n]+=1
        if(out==1):
            succ[n]+=1
        else:
            fail[n]+=1
        pval = np.random.beta(succ+1,fail+1)
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg

def ucb_t2(hz,ins,c):
    l = len(ins)
    num = np.ones(l)
    reward = np.zeros(l)
    avg = np.zeros(l)
    for k in range(l):
        out = pull_arm(ins,k)
        reward[k]+=out
    avg = reward/num
    ucb=np.copy(avg)

    for t in range(l,hz):
        mucb=np.max(ucb)
        nmaxs=[k for k in range(l) if ucb[k]==mucb]#tiebreaker
        n=np.random.choice(nmaxs)
        out = pull_arm(ins,n)
        reward[n]+=out
        num[n]+=1
        avg = reward/num
        ucb = avg + np.sqrt(c*np.log(t)/num)
    rew = np.sum(reward)
    mcr = hz*np.max(ins)
    reg = mcr-rew
    return reg

def alg_t3(ins,hz):
    l = len(ins)-1
    num = np.ones(l)
    reward = np.zeros(l)
    succ = np.zeros(l)
    fail = np.zeros(l)
    pval = np.random.beta(succ+1,fail+1)
    for t in range(0,hz):
        mpv = np.max(pval)
        nmaxs=[k for k in range(l) if pval[k]==mpv]#tiebreaker
        n=np.random.choice(nmaxs)
        out = pull_arm_t34(ins,n)
        reward[n]+=out
        num[n]+=1
        succ[n]+=out
        fail[n]+=1-out
        pval = np.random.beta(succ+1,fail+1)
    rew = np.sum(reward)
    em = []
    for i in range(1,l+1):
        em.append(np.dot(ins[0],ins[i]))
    mcr = hz*np.max(em)
    reg = mcr-rew
    return reg

def alg_t4(ins,hz,th):
    l = len(ins)-1
    num = np.ones(l)
    reward = np.zeros(l)
    succ = np.zeros(l)
    fail = np.zeros(l)
    pval = np.random.beta(succ+1,fail+1)
    for t in range(0,hz):
        mpv = np.max(pval)
        nmaxs=[k for k in range(l) if pval[k]==mpv]#tiebreaker
        n=np.random.choice(nmaxs)
        out = pull_arm_t34(ins,n)
        reward[n]+=out
        num[n]+=1
        if(out>th):
            succ[n]+=1
        else:
            fail[n]+=1
        pval = np.random.beta(succ+1,fail+1)
    rew = np.sum(succ)
    em = []
    for i in range(1,l+1):
        q = [ins[i][p] for p in range(len(ins[0])) if ins[0][p]>th]
        em.append(np.sum(q))
    mcr = hz*np.max(em)
    reg = mcr-rew
    return reg,rew

        
if __name__ == "__main__":
    
    if(al=="epsilon-greedy-t1"):
        ins = openfile(insta)
        reg = epsilong3(ep,hz,ins)
        print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    if(al=="ucb-t1"):
        ins = openfile(insta)
        reg = ucbf(ins,hz)
        print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    if(al=="kl-ucb-t1"):
        ins = openfile(insta)
        reg = klucbf(ins,hz)
        print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    if(al=="thompson-sampling-t1"):
        ins = openfile(insta)
        reg = thompson(ins,hz)
        print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    if(al=="ucb-t2"):
        ins = openfile(insta)
        reg = ucb_t2(hz,ins,c)
        print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    if(al=="alg-t3"):
        ins = openfile_t34(insta)
        reg = alg_t3(ins,hz)
        print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    if(al=="alg-t4"):
        ins = openfile_t34(insta)
        reg,high = alg_t4(ins,hz,th)
        print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", high, sep='')
