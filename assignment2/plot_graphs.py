import pandas as pd
import glob
import matplotlib.pyplot as plt
import numpy as np

########################################################################
#                                                                      #
#                       4.1 & 4.2                                      #
#                                                                      #
########################################################################

allFiles = []
list_=[]

path = r"C:\Users\Belmir\Desktop\Masters\Winter_2019\IFT6135\Assignment_2\q4.1_4.2"
for f in glob.iglob(path+"/*/*og.txt"):
    allFiles.append(f)
 

for file_ in allFiles:
    df=pd.read_csv(file_, delim_whitespace=True,header=None)
    time =np.asarray(df.iloc[:,16])
    time = np.cumsum(time)
    df['17'] = time
    list_.append(df)
    
#################################################################################Plot 4.1#
##### PPL X EPOCHS
'''plt.figure()
plt.plot(list_[1][1],list_[1][4] ,label='Train - GRU - LRSchedule',marker="*")
plt.plot(list_[1][1],list_[1][7], label='Validation - GRU - LRSchedule',marker="*")
plt.plot(list_[3][1],list_[3][4] ,label='Train - RNN - ADAM',marker="o")
plt.plot(list_[3][1],list_[3][7], label='Validation - RNN - ADAM',marker="o")
plt.plot(list_[7][1],list_[7][4] ,label='Train - Transformer - LRSchedule',marker="^")
plt.plot(list_[7][1],list_[7][7], label='Validation - Transformer - LRSchedule',marker="^")
plt.xlabel('Epochs')
plt.ylabel('PPL')
plt.title("PPL x Epochs")
plt.legend()
plt.ylim(0,800)
plt.xlim(0,39)
plt.show()

##### PPL X  Wall Clock Time
plt.figure()
plt.plot(list_[1]['17'],list_[1][4] , label='Train - GRU - LRSchedule',marker="*")
plt.plot(list_[1]['17'],list_[1][7] , label='Validation - GRU - LRSchedule',marker="*")
plt.plot(list_[3]['17'],list_[3][4] , label='Train - RNN - ADAM',marker="o")
plt.plot(list_[3]['17'],list_[3][7] , label='Validation - RNN - ADAM',marker="o")
plt.plot(list_[7]['17'],list_[7][4] , label='Train - Transformer - LRSchedule',marker="^")
plt.plot(list_[7]['17'],list_[7][7] , label='Validation - Transformer - LRSchedule',marker="^")
plt.xlabel('Wall Clock Time')
plt.ylabel('PPL')
plt.title("PPL x Wall Clock Time")
plt.legend()
plt.ylim(0,800)
plt.xlim(0,7000)
plt.show()

##############################################################################################Plot 4.2#

##### Optimizers x ADAM
plt.figure()
plt.plot(list_[0][1],list_[0][4] ,label='Train - GRU',marker="*")
plt.plot(list_[0][1],list_[0][7], label='Validation - GRU',marker="*")
plt.plot(list_[3][1],list_[3][4] ,label='Train - RNN',marker="o")
plt.plot(list_[3][1],list_[3][7], label='Validation - RNN',marker="o")
plt.plot(list_[6][1],list_[6][4] ,label='Train - Transformer',marker="^")
plt.plot(list_[6][1],list_[6][7], label='Validation - Transformer',marker="^")
plt.xlabel('Epochs')
plt.ylabel('PPL')
plt.title("PPL x Epochs - ADAM")
plt.legend()
plt.ylim(0,800)
plt.xlim(0,39)
plt.show()

plt.figure()
plt.plot(list_[0]['17'],list_[0][4] , label='Train - GRU',marker="*")
plt.plot(list_[0]['17'],list_[0][7] , label='Validation - GRU',marker="*")
plt.plot(list_[3]['17'],list_[3][4] , label='Train - RNN',marker="o")
plt.plot(list_[3]['17'],list_[3][7] , label='Validation - RNN',marker="o")
plt.plot(list_[6]['17'],list_[6][4] , label='Train - Transformer',marker="^")
plt.plot(list_[6]['17'],list_[6][7] , label='Validation - Transformer',marker="^")
plt.xlabel('Wall Clock Time')
plt.ylabel('PPL')
plt.title("PPL x Wall Clock Time - ADAM")
plt.legend()
plt.ylim(0,800)
plt.show()

##### Optimizers x LRSchedule
plt.figure()
plt.plot(list_[1][1],list_[1][4] ,label='Train - GRU',marker="*")
plt.plot(list_[1][1],list_[1][7], label='Validation - GRU',marker="*")
plt.plot(list_[4][1],list_[4][4] ,label='Train - RNN',marker="o")
plt.plot(list_[4][1],list_[4][7], label='Validation - RNN',marker="o")
plt.plot(list_[7][1],list_[7][4] ,label='Train - Transformer',marker="^")
plt.plot(list_[7][1],list_[7][7], label='Validation- Transformer',marker="^")
plt.xlabel('Epochs')
plt.ylabel('PPL')
plt.title("PPL x Epochs - SGD-LRSchedule")
plt.legend()
plt.ylim(0,800)
plt.xlim(0,39)
plt.show()

plt.figure()
plt.plot(list_[1]['17'],list_[1][4] , label='Train - GRU',marker="*")
plt.plot(list_[1]['17'],list_[1][7] , label='Validation - GRU',marker="*")
plt.plot(list_[4]['17'],list_[4][4] , label='Train - RNN',marker="o")
plt.plot(list_[4]['17'],list_[4][7] , label='Validation - RNN',marker="o")
plt.plot(list_[7]['17'],list_[7][4] , label='Train - Transformer',marker="^")
plt.plot(list_[7]['17'],list_[7][7] , label='Validation - Transformer',marker="^")
plt.xlabel('Wall Clock Time')
plt.ylabel('PPL')
plt.title("PPL x Wall Clock Time - SGD-LRSchedule")
plt.legend()
plt.ylim(0,800)
plt.show()

##### Optimizers x SGD

plt.figure()
plt.plot(list_[2][1],list_[2][4] ,label='Train - GRU',marker="*")
plt.plot(list_[2][1],list_[2][7], label='Validation - GRU',marker="*")
plt.plot(list_[5][1],list_[5][4] ,label='Train - RNN',marker="o")
plt.plot(list_[5][1],list_[5][7], label='Validation - RNN',marker="o")
plt.plot(list_[8][1],list_[8][4] ,label='Train - Transformer',marker="^")
plt.plot(list_[8][1],list_[8][7], label='Validation - Transformer',marker="^")
plt.xlabel('Epochs')
plt.ylabel('PPL')
plt.title("PPL x Epochs - SGD")
plt.legend()
plt.ylim(0,10000)
plt.xlim(0,39)
plt.show()

plt.figure()
plt.plot(list_[2]['17'],list_[2][4] , label='Train - GRU',marker="*")
plt.plot(list_[2]['17'],list_[2][7] , label='Validation - GRU',marker="*")
plt.plot(list_[5]['17'],list_[5][4] , label='Train - RNN',marker="o")
plt.plot(list_[5]['17'],list_[5][7] , label='Validation - RNN',marker="o")
plt.plot(list_[8]['17'],list_[8][4] , label='Train - Transformer',marker="^")
plt.plot(list_[8]['17'],list_[8][7] , label='Validation - Transformer',marker="^")
plt.xlabel('Wall Clock Time')
plt.ylabel('PPL')
plt.title("PPL x Wall Clock Time - SGD")
plt.legend()
plt.ylim(0,10000)
plt.show()'''

########################################################################
#                                                                      #
#                       4.1 & 4.2                                      #
#                                                                      #
########################################################################
allFiles = []
list_=[]

path = r"C:\Users\Belmir\Desktop\Masters\Winter_2019\IFT6135\Assignment_2\Best_GRU"
for f in glob.iglob(path+"/*/*og.txt"):
    allFiles.append(f)
 

for file_ in allFiles:
    df=pd.read_csv(file_, delim_whitespace=True,header=None)
    time =np.asarray(df.iloc[:,16])
    time = np.cumsum(time)
    df['17'] = time
    list_.append(df)

plt.figure()
color = ['b','g','r','c','m','y']
for i in range(len(list_)):
    plt.plot(list_[i][1],list_[i][4] ,label='Train model '+str(i+1),marker="o",color=color[i])
    plt.plot(list_[i][1],list_[i][7], label='Validation model '+str(i+1),marker="^",color=color[i])
plt.xlabel('Epochs')
plt.ylabel('PPL')
plt.title("PPL x Epochs - GRU")
plt.legend()
plt.ylim(0,800)
plt.xlim(0,39)
plt.show()

##### PPL X  Wall Clock Time
plt.figure()
color = ['b','g','r','c','m','y']
for i in range(len(list_)):
    plt.plot(list_[i]['17'],list_[i][4] ,label='Train model '+str(i+1),marker="o",color=color[i])
    plt.plot(list_[i]['17'],list_[i][7], label='Validation model '+str(i+1),marker="^",color=color[i])
plt.xlabel('Wall Clock Time')
plt.ylabel('PPL')
plt.title("PPL x Wall Clock Time - GRU")
plt.legend()
plt.ylim(0,800)
plt.show()


allFiles = []
list_=[]

path = r"C:\Users\Belmir\Desktop\Masters\Winter_2019\IFT6135\Assignment_2\Best_RNN"
for f in glob.iglob(path+"/*/*og.txt"):
    allFiles.append(f)
 

for file_ in allFiles:
    df=pd.read_csv(file_, delim_whitespace=True,header=None)
    time =np.asarray(df.iloc[:,16])
    time = np.cumsum(time)
    df['17'] = time
    list_.append(df)

plt.figure()
color = ['b','g','r','c','m','y']
for i in range(len(list_)):
    plt.plot(list_[i][1],list_[i][4] ,label='Train model '+str(i+1),marker="o",color=color[i])
    plt.plot(list_[i][1],list_[i][7], label='Validation model '+str(i+1),marker="^",color=color[i])
plt.xlabel('Epochs')
plt.ylabel('PPL')
plt.title("PPL x Epochs - RNN")
plt.legend()
plt.ylim(0,800)
plt.xlim(0,39)
plt.show()

##### PPL X  Wall Clock Time
plt.figure()
color = ['b','g','r','c','m','y']
for i in range(len(list_)):
    plt.plot(list_[i]['17'],list_[i][4] ,label='Train model '+str(i+1),marker="o",color=color[i])
    plt.plot(list_[i]['17'],list_[i][7], label='Validation model '+str(i+1),marker="^",color=color[i])
plt.xlabel('Wall Clock Time')
plt.ylabel('PPL')
plt.title("PPL x Wall Clock Time - RNN")
plt.legend()
plt.ylim(0,800)
plt.show()

allFiles = []
list_=[]

path = r"C:\Users\Belmir\Desktop\Masters\Winter_2019\IFT6135\Assignment_2\Best_Transformer"
for f in glob.iglob(path+"/*/*og.txt"):
    allFiles.append(f)
 

for file_ in allFiles:
    df=pd.read_csv(file_, delim_whitespace=True,header=None)
    time =np.asarray(df.iloc[:,16])
    time = np.cumsum(time)
    df['17'] = time
    list_.append(df)

plt.figure()
color = ['b','g','r','c','m','y']
for i in range(len(list_)):
    plt.plot(list_[i][1],list_[i][4] ,label='Train model '+str(i+1),marker="o",color=color[i])
    plt.plot(list_[i][1],list_[i][7], label='Validation model '+str(i+1),marker="^",color=color[i])
plt.xlabel('Epochs')
plt.ylabel('PPL')
plt.title("PPL x Epochs - Transformer")
plt.legend()
plt.ylim(0,800)
plt.xlim(0,39)
plt.show()

##### PPL X  Wall Clock Time
plt.figure()
color = ['b','g','r','c','m','y']
for i in range(len(list_)):
    plt.plot(list_[i]['17'],list_[i][4] ,label='Train model '+str(i+1),marker="o",color=color[i])
    plt.plot(list_[i]['17'],list_[i][7], label='Validation model '+str(i+1),marker="^",color=color[i])
plt.xlabel('Wall Clock Time')
plt.ylabel('PPL')
plt.title("PPL x Wall Clock Time - Transformer")
plt.legend()
plt.ylim(0,800)
plt.show()

