import sys
import numpy as np
import matplotlib.pyplot as plt
import warnings
# warnings.filterwarnings("ignore")

# ins = "../instances/instances-task1/i-1.txt"
# f2 = "/home/atharva/cs747/cs747-pa1-v1/instances/instances-task1/i-1.txt"
# f3 = "/home/atharva/cs747/cs747-pa1-v1/instances/instances-task1/i-1.txt"

# insta = sys.argv[2]
# al = sys.argv[4]
# rs = int(sys.argv[6])
# ep = float(sys.argv[8])
# c = float(sys.argv[10])
# th = float(sys.argv[12])
# hz = int(sys.argv[14])

def openfile(ins):
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

def pull_arm(ins,n):
    sup = ins[0]
    prob = ins[n+1]
    op = np.random.choice(sup,1, p=prob)
    return op


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
        out = pull_arm(ins,n)
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
        out = pull_arm(ins,n)
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
    # ins = openfile(insta)
    # if(al=="epsilon-greedy-t1"):
    #     reg = epsilong3(ep,hz,ins)
    #     print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    # if(al=="ucb-t1"):
    #     reg = ucbf(ins,hz)
    #     print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    # if(al=="kl-ucb-t1"):
    #     reg = klucbf(ins,hz)
    #     print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    # if(al=="thompson-sampling-t1"):
    #     reg = thompson(ins,hz)
    #     print(insta,", ", al,", ", rs,", ", ep,", ", c,", ", th,", ", hz,", ", reg,", ", 0, sep='')
    
    instances = ["../instances/instances-task4/i-1.txt", "../instances/instances-task4/i-2.txt"]
    algorithms = ["alg-t4"]
    threshold = [0.2,0.6]
    horizons = [100,400,1600,6400,25600,102400]
    filew = open("output_t4.txt","a")
    i = 1
    for insta in instances:
        print("ins{}".format(i))
        ins = openfile(insta)
        for th in threshold:
            plt.figure()
            plt.xlabel("Horizon")
            plt.ylabel("Regret")
            # name = ["epsilon-greedy","ucb","kl-ucb","thompson-sampling"]
            for al in algorithms:
                print(al)
                mregrs = []
                for hz in horizons:
                    regrs = []
                    for rs in range(50):
                        np.random.seed(rs)
                        if(al=="alg-t4"):
                            reg,high = alg_t4(ins,hz,th)
                            filew.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(insta, al, rs, 0.02, 2, th, hz, reg, high))
                        regrs.append(reg)
                    regrs = np.array(regrs)
                    mreg = np.mean(regrs)
                    mregrs.append(mreg)
                plt.scatter(horizons, mregrs)
                plt.plot(horizons, mregrs)
            plt.xscale("log")
            plt.savefig('ins{}'.format(i))
            i+=1
