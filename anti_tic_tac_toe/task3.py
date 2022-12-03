import os
import numpy as np
state_p1 = 'data/attt/states/states_file_p1.txt'
state_p2 = 'data/attt/states/states_file_p2.txt'

po = 'data/attt/policies/p2_policy1.txt'

pol_list = [po]

for i in range(0,20,2):
    os.system('python encoder.py --policy {} --states {} > mdpfile.txt'.format(pol_list[i], state_p1))
    os.system('python planner.py --mdp mdpfile.txt > valpol.txt')
    os.system('python decoder.py --value-policy valpol.txt --states {} --player-id 1 > task3/p1_policyfile_{}.txt'.format(state_p1, i))
    os.remove('mdpfile.txt')
    os.remove('valpol.txt')
    pol_list.append('task3/p1_policyfile_{}.txt'.format(i))
    os.system('python encoder.py --policy {} --states {} > mdpfile.txt'.format(pol_list[i+1], state_p2))
    os.system('python planner.py --mdp mdpfile.txt > valpol.txt')
    os.system('python decoder.py --value-policy valpol.txt --states {} --player-id 2 > task3/p2_policyfile_{}.txt'.format(state_p2, i))
    os.remove('mdpfile.txt')
    os.remove('valpol.txt')
    pol_list.append('task3/p2_policyfile_{}.txt'.format(i))



def get_norm(i1,i2):
    file = open(i1,"r")
    lines = file.readlines()
    action = []
    states = []
    lines.pop(0)
    for line in lines:
        line = line.split()
        states.append(line[0])
        probs = []
        for i in range(1,len(line)):
            probs.append(float(line[i]))
        action.append(probs)
    file.close()
    pol1 = np.zeros(len(states))
    for i in range(len(states)):
        pol1[i] = action[i].index(1)

    file = open(i2,"r")
    lines = file.readlines()
    action = []
    states = []
    lines.pop(0)
    for line in lines:
        line = line.split()
        states.append(line[0])
        probs = []
        for i in range(1,len(line)):
            probs.append(float(line[i]))
        action.append(probs)
    file.close()
    pol2 = np.zeros(len(states))
    for i in range(len(states)):
        pol2[i] = action[i].index(1)

    diff = pol1-pol2
    return np.linalg.norm(diff)

print(get_norm(pol_list[-2],pol_list[-4]), 'l2 norm of last 2 policies for player 1')
print(get_norm(pol_list[-1],pol_list[-3]), 'l2 norm of last 2 policies for player 2')

print('If the norm is zero, this means that the sequence of policies generated for each player guaranteed to converge!!')