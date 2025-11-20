import pandas as pd
import numpy as np


data = pd.read_csv(r'C:\Users\acer\Downloads\P3Data.csv')
print(data)
concepts = np.array(data)[:,:-1]
print(concepts)
target = np.array(data)[:,-1]
print(target)
def learn(concepts, target):
    print('Initialisation of specific_h and general_h\n')
    specific_h = concepts[0].copy()
    print('\nSpecific hypothesis:',specific_h)
    general_h = [['?' for i in range(len(specific_h))]for i in range(len(specific_h))]
    print('General hypothesis:',general_h)
    for i,h in enumerate(concepts):
        print('\nInstance',i+1, 'is', h)
        if target[i] == 'Yes':
            print('\nInstance is positive')
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == 'No':
            print('\nInstance is negative')
            for x in range(len(specific_h)):
                 if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                 else:
                    general_h[x][x] = '?'
        print('\nSpecific_h after',i+1,'instance is', specific_h)
        print('\nGeneral_h after',i+1,'instance is', general_h)
    indices = [i for i, val in enumerate(general_h) if val == ['?','?','?','?','?','?']]
    for i in indices:
            general_h.remove(['?','?','?','?','?','?'])
    return specific_h,general_h

s_final, g_final = learn(concepts, target)
print("Final Specific_h: ", s_final, sep="\n")
print("Final General_h: ", g_final, sep="\n")

