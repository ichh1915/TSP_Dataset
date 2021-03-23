

import numpy as np
import math  
import torch
from torch.utils import data
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        x = self.linear(x)

        return x
    
    
class MyDataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self,index):
        X = self.X[index,:].float()
        Y = self.Y[index].float()
   
        return X,Y
    
""""""""""""""""""""""""""" Read Data & Initialize Hyper-parameters """""""""""""""""""""""""""
#0.6277 ≤ β ≤ 0.922



# [A_est_PCA, graph.A_sub, A_Full, lambda_1_S_Sub, lambda_2_S_Sub, lambda_1_S_Full, lambda_2_S_Full]2

x_Data_test = np.load('Output/Beijing_S_2800_X_PCA_test.npy')    
y_Data_test = np.load('Output/Beijing_S_2800_Y_PCA_test.npy')   


city = 3
if city == 0:
    #x_Data = np.load('Output/Manhattan_2800_X_PCA.npy')    
    #y_Data = np.load('Output/Manhattan_2800_Y_PCA.npy')  
    
    x_Data = np.load('Output/Manhattan_S_2800_X_PCA_SV.npy')    
    y_Data = np.load('Output/Manhattan_S_2800_Y_PCA_SV.npy')      


elif city == 1:
    #x_Data = np.load('Output/Beijing_S_2800_X_PCA.npy')    
    #y_Data = np.load('Output/Beijing_S_2800_Y_PCA.npy')   
    
    x_Data = np.load('Output/Beijing_S_2800_X_PCA_SV.npy')    
    y_Data = np.load('Output/Beijing_S_2800_Y_PCA_SV.npy')      

    y_Data_TEST = np.load('Output/Beijing_2800_Y_PCA.npy')   
    

elif city == 2:
    #x_Data = np.load('Output/Beijing_S_2800_X_PCA.npy')    
    #y_Data = np.load('Output/Beijing_S_2800_Y_PCA.npy')   
    
    x_Data = np.load('Output/Beijing_5R_2800_X_PCA_SV.npy')    
    y_Data = np.load('Output/Beijing_5R_2800_Y_PCA_SV.npy')      

    


elif city == 3:
    #x_Data = np.load('Output/London_2800_X_PCA.npy')    
    #y_Data = np.load('Output/London_2800_Y_PCA.npy')   
    
    x_Data = np.load('Output/London_S_2800_X_PCA_SV.npy')    
    y_Data = np.load('Output/London_S_2800_Y_PCA_SV.npy')      



y_sum_TS = np.sum(y_Data_test)    
y_sum_GLS = np.sum(y_Data)    

#x_Data = x_Data[:1600,:]
#y_Data = y_Data[:1600]

x_Data = x_Data[:2800]
y_Data = y_Data[:2800]
A_pca = x_Data[:,0]
A_pca = np.expand_dims(A_pca, axis=1)


"""
A_sub = x_Data[:,1]
A_sub = np.expand_dims(A_sub, axis=1)

A_full = x_Data[:,2]
A_full = np.expand_dims(A_full, axis=1)

lambda_1_G_Sub = x_Data[:,3]

lambda_2_G_Sub = x_Data[:,4]


lambda_1_G_Sub = np.expand_dims(lambda_1_G_Sub, axis=1)
lambda_2_G_Sub = np.expand_dims(lambda_2_G_Sub, axis=1)

lambda_avg = 2 * np.sqrt(lambda_1_G_Sub*lambda_2_G_Sub)

lambda_1_2_G_Sub = x_Data[:,3:5]
lambda_1_2_G_Full = x_Data[:,5:7]


test = (A_sub - lambda_1_G_Sub)/A_sub
"""
""""""""""""""""""""""""""" Print Data """""""""""""""""""""""""""

"""
plt.figure()

plt.scatter(A_sub, y_Data, s=2)

plt.xlabel('sqrt(n * A)')
plt.ylabel('y')
plt.legend()

plt.figure()

plt.scatter(lambda_1_G_Sub , y_Data, s=2)

plt.xlabel('sqrt(n * lmabda_1_(G))')
plt.ylabel('y')
plt.legend()

plt.figure()

plt.scatter(lambda_2_G_Sub , y_Data, s=2)

plt.xlabel('sqrt(n * lmabda_2_(G))')
plt.ylabel('y')
plt.legend()
"""





""""""""""""""""""""""""""" Train Model: Lambda_G """""""""""""""""""""""""""



data_size = np.shape(x_Data)[0]


c_Data = np.ones((data_size,1))
#x_Train = np.concatenate((A_pca, c_Data),axis=1)    
x_Train = A_pca
y_Data = np.array(y_Data,dtype=np.float32)


model = LinearRegression(fit_intercept=True)

model.fit(x_Train, y_Data)


weights = model.coef_
constant = model.intercept_



print('weights',weights)
print('constant',constant)


y_pred = model.predict(x_Train)
y_true = y_Data

plt.figure()

plt.scatter(x_Train[:,0] , y_pred, s=2)
plt.scatter(x_Train[:,0] , y_true, color = 'r', s=2)

plt.title('Beta = '+str(weights[0]))
plt.xlabel('sqrt(n * A_pca)')
plt.ylabel('y')
plt.legend()
plt.xlim([0,x_Train.max()])
plt.ylim([0,y_true.max()])

plt.figure()

"""

plt.scatter(x_Train[:,1] , y_pred, s=2)
plt.scatter(x_Train[:,1] , y_true, color = 'r', s=2)

plt.title('Beta = '+str(weights[1]))
plt.xlabel('sqrt(n * lambda_2_G)')
plt.ylabel('y')
plt.legend()
"""

MSE = mean_squared_error(y_true, y_pred)
print('RMSE', np.sqrt(MSE))

""""""""""""""""""""""""""" Train Model: A """""""""""""""""""""""""""

"""
data_size = np.shape(x_Data)[0]


c_Data = np.ones((data_size,1))
x_Train = np.concatenate((A_sub,c_Data),axis=1)    
    
y_Data = np.array(y_Data,dtype=np.float32)


model = LinearRegression()

model.fit(x_Train, y_Data)


weights = model.coef_




y_pred = model.predict(x_Train)
y_true = y_Data
plt.figure()

plt.scatter(x_Train[:,0] , y_pred, s=2)
plt.scatter(x_Train[:,0] , y_true, color = 'r', s=2)

plt.title('Beta = '+str(weights[0]))
plt.xlabel('sqrt(n * A)')
plt.ylabel('y')
plt.legend()
MSE = mean_squared_error(y_true, y_pred)
print('RMSE', np.sqrt(MSE))


"""


Size_lst = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 80] # 40, 50, 60, 80]
N_Per_Size = 200 # 200

G_size = np.ones((data_size))
i = 0
for size in Size_lst:
    for n in range(N_Per_Size):
        G_size[i] = size
        i += 1




plt.figure()

plt.errorbar(G_size,y_pred,fmt='.k')
plt.ylabel('Estimated Tour Length')
plt.xlabel('Number of Nodes: TSP')

plt.figure()
yerr = np.absolute(y_true - y_pred)/y_true

plt.errorbar(G_size,yerr,fmt='.k')
plt.ylabel('Estimated Error')
plt.xlabel('Number of Nodes: TSP')

Var_lst = []
RMSE_lst = []
Coeff_Var_lst = []
Mean_lst = []
for size in Size_lst:
    Var_lst.append( np.var(y_pred[G_size==size]) )
    RMSE_lst.append( np.mean(yerr[G_size==size]) )
    #Mean_lst.append( np.mean(y_true[G_size==size]) )
    
    Mean_lst.append( np.mean(y_pred[G_size==size]) )

    Coeff_Var_lst.append( np.std(y_pred[G_size==size])/np.mean(y_pred[G_size==size]) )

plt.figure()

plt.plot(Size_lst,Var_lst,marker='^')
plt.ylabel('Var[Estimated Tour Length] (m^2)')
plt.xlabel('Number of Nodes: TSP')


plt.figure()


plt.plot(Size_lst,np.array(RMSE_lst)*100,marker='^')
plt.ylabel('Optimality Gap (%)')
plt.xlabel('Number of Nodes: TSP')




plt.figure()

plt.plot(Size_lst,Coeff_Var_lst,marker='^')
plt.ylabel('Coefficient of variation[Estimated Tour Length]')
plt.xlabel('Number of Nodes: TSP')




"""
plt.figure(figsize=(10,8))

plt.plot(Size_lst,RMSE_lst,marker='^')
plt.ylabel('Absolute Estimation Error')
plt.xlabel('Number of Nodes: TSP')
"""

plt.figure()


plt.plot(Size_lst,Mean_lst,marker='^')
plt.ylabel('Expectation[TSP Tour Length by BHH model]')
plt.xlabel('N')

tst = np.sqrt(Size_lst)


plt.figure()

plt.plot(np.sqrt(Size_lst),Mean_lst,marker='^')
plt.ylabel('Expectation[TSP Tour Length]')
plt.xlabel('Sqrt(N)')


"""
Mean_lst = []
for size in Size_lst:
    Var_lst.append( np.var(y_pred[G_size==size]) )
    RMSE_lst.append( np.mean(yerr[G_size==size]) )
    Mean_lst.append( np.mean(y_true[G_size==size]) )
    
    #Mean_lst.append( np.mean(y_pred[G_size==size]) )

    Coeff_Var_lst.append( np.std(y_pred[G_size==size])/np.mean(y_pred[G_size==size]) )

plt.plot(np.sqrt(Size_lst),Mean_lst,marker='^')


betas = (y_Data-constant)/x_Data[:,0]


plt.figure(figsize = (5,10))
plt.boxplot(betas,whis=20) 
plt.title('Box plot of beta')
plt.yticks(np.arange(min(betas), max(betas), 2))

plt.figure(figsize = (10,6))
arr = plt.hist(betas, bins=1000)
plt.xlabel('Beta values')
plt.ylabel('Count')
plt.xticks(np.arange(min(betas), max(betas), 2))
plt.title('Histogram of Beta, argmax_Beta=0.6834')
"""



