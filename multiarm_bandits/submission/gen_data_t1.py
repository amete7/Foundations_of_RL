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
        line = line.rstrip('\n')
        p = float(line)
        inst.append(p)
    file.close()
    instance = np.array(inst)
    return instance

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

def ucb_t2(ins,hz,c):
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
    
    instances = ["../instances/instances-task1/i-1.txt", "../instances/instances-task1/i-2.txt", "../instances/instances-task1/i-3.txt"]
    algorithms = ["epsilon-greedy-t1", "ucb-t1", "kl-ucb-t1", "thompson-sampling-t1"]
    horizons = [100,400,1600,6400,25600,102400]
    ep = 0.02
    filew = open("output_t1.txt","a")
    i = 1
    for insta in instances:
        print("ins{}".format(i))
        j=0
        ins = openfile(insta)
        plt.figure()
        plt.xlabel("Horizon")
        plt.ylabel("Regret")
        name = ["epsilon-greedy","ucb","kl-ucb","thompson-sampling"]
        for al in algorithms:
            print(al)
            mregrs = []
            for hz in horizons:
                regrs = []
                for rs in range(50):
                    np.random.seed(rs)
                    if(al=="epsilon-greedy-t1"):
                        reg = epsilong3(ep,hz,ins)
                        filew.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(insta, al, rs, ep, 2, 0, hz, reg, 0))
                    if(al=="ucb-t1"):
                        reg = ucbf(ins,hz)
                        filew.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(insta, al, rs, ep, 2, 0, hz, reg, 0))
                    if(al=="kl-ucb-t1"):
                        reg = klucbf(ins,hz)
                        filew.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(insta, al, rs, ep, 2, 0, hz, reg, 0))
                    if(al=="thompson-sampling-t1"):
                        reg = thompson(ins,hz)
                        filew.write("{}, {}, {}, {}, {}, {}, {}, {}, {}\n".format(insta, al, rs, ep, 2, 0, hz, reg, 0))
                    regrs.append(reg)
                regrs = np.array(regrs)
                mreg = np.mean(regrs)
                mregrs.append(mreg)
            plt.scatter(horizons, mregrs)
            plt.plot(horizons, mregrs, label=name[j])
            j+=1
        plt.xscale("log")
        plt.legend(loc="upper left")
        plt.savefig('ins{}'.format(i))
        i+=1
        # plt.show()