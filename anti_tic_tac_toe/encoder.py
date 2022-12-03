import sys
import numpy as np
from numpy.core.defchararray import index

pfile = sys.argv[2]
sfile = sys.argv[4]

astates = []
estates = []
action = []
file = open(sfile,"r")
lines = file.readlines()
for line in lines:
    line = line.split()
    astates.append(line[0])
file.close()

file = open(pfile,"r")
lines = file.readlines()
opagent = int(lines[0][0])
agent = 1 if int(lines[0][0])==2 else 2
# print(opagent,"agebt")
lines.pop(0)
for line in lines:
    line = line.split()
    # print(line)
    estates.append(line[0])
    probs = []
    for i in range(1,len(line)):
        probs.append(float(line[i]))
    action.append(probs)
file.close()

def get_reward(st):
    lst = list(st)
    lst = [int(a) for a in lst]
    if lst[0]==lst[1]==lst[2]==agent:
        return 0
    if lst[3]==lst[4]==lst[5]==agent:
        return 0
    if lst[6]==lst[7]==lst[8]==agent:
        return 0
    if lst[0]==lst[3]==lst[6]==agent:
        return 0
    if lst[1]==lst[4]==lst[7]==agent:
        return 0
    if lst[2]==lst[5]==lst[8]==agent:
        return 0
    if lst[0]==lst[4]==lst[8]==agent:
        return 0
    if lst[2]==lst[4]==lst[6]==agent:
        return 0
    if all(i != 0 for i in lst):
        return 0
    else:
      return 1


if __name__ == "__main__":
    trans = []
    c1 = 0
    c2 = 0
    new_state = astates.copy()
    for state in astates:
        pa = [a for a in range(len(state)) if state[a]=="0"]
        for a in pa:
            lst = list(state)
            lst[a] = str(agent)
            se = "".join(lst)
            if se not in estates:
                if se not in new_state:
                    new_state.append(se)
                    c1+=1
                reward = get_reward(se)
                trans.append([new_state.index(state),a,new_state.index(se),1,reward])
            else:
                pad = [b for b in range(len(se)) if se[b]=="0"]
                for b in pad:
                    trprob = action[estates.index(se)][b]
                    if(trprob>1e-8):
                        lst = list(se)
                        lst[b] = str(opagent)
                        sd = "".join(lst)
                        if sd not in astates:
                            if sd not in new_state:
                                new_state.append(sd)
                                c2+=1
                            reward = get_reward(sd)
                        else:
                            reward = 0
                        trans.append([new_state.index(state),a,new_state.index(sd),trprob,reward])
    print('numStates', len(new_state))
    print('numAction', 9)
    print('end', end=' ')
    for i in range(len(astates),len(new_state)-1):
        print(i, end=' ')
    print(len(new_state)-1)
    for i in range(len(trans)):
        print('transition', trans[i][0], trans[i][1], trans[i][2], trans[i][4], trans[i][3])
    print('mdptype', 'episodic')
    print('discount', 0.9)