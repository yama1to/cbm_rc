# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc　時系列生成タスク　
cbmrc6e.pyを改変
Configクラスによるパラメータ設定
"""

import argparse
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import time
from explorer import common
from generate_data_sequence_approximation import generate_data
from generate_matrix import *
from test_cbm_esn_approximation2 import Config2
#from test_esn_approximation2 import Config



class Config:
    def __init__(self) -> None:
        pass

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr")
    Wb = generate_random_matrix(c.Nh,c.Ny,c.alpha_b,c.beta_b,distribution="one",normalization="none")
    Wi = generate_random_matrix(c.Nh,c.Nu,c.alpha_i,c.beta_i,distribution="one",normalization="none")
    Wo = np.zeros(c.Nh * c.Ny).reshape((c.Ny, c.Nh))

def fy(h):
    return np.tanh(h)

def run_network(mode):
    global Hp
    
    Hp = np.zeros((c.MM, c.Nh))
    #x = np.random.uniform(-1, 1, Nh)/ 10**4
    x = np.zeros(c.Nh)
    

    for n in range(c.MM):
        
        u = Up[n, :]

        #Hp[n+1,:] = x + 1.0/tau * (-alpha0 * x + fx(Wi@u + Wr@x))
        next_x = (1 - c.alpha0) * x + c.alpha0*fy(Wi@u + Wr@x)
        Hp[n,:] = next_x
        x= next_x

        


def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = Hp[c.MM0:, :]
    invD = Dp
    G = invD[c.MM0:, :]

    #print("Hp\n",Hp)
    #print("M\n",M)

    ### Ridge regression
    if c.lambda0 == 0:
        Wo = np.dot(G.T,np.linalg.pinv(M).T)
        #print("a")
    else:
        E = np.identity(c.Nh)
        TMP1 = np.linalg.inv(M.T@M + c.lambda0 * E)
        WoT = TMP1@M.T@G
        Wo = WoT.T

    #print("WoT\n", WoT)

def test_network():
    global Yp
    run_network(0)

    YpT = Wo @ Hp.T
    Yp = YpT.T

def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,Yp
    global RMSE1,RMSE2

    c.Nh = int(c.Nh)

    np.random.seed(seed = int(c.seed))    
    generate_weight_matrix()

    ### generate data
    
 
    #print(c.MM,c.delay,c.logv)
    U,D = generate_data(num=c.MM,delay=c.delay,logv=c.logv, f=c.f)

    ### training
    #print("training...")
    
    #Scale to (-1,1)
    Dp = D
    Up = U
    train_network()
    #print("...end") 
    
    ### test
    #print("test...")
    test_network()                  #OUTPUT = Yp


    ### evaluation
    sum=0
    c.MM0 = 10
    for j in range(c.MM0,c.MM-1):
        sum += (Yp[j] - Dp[j])**2

    SUM=np.sum(sum)
    RMSE1 = np.sqrt(SUM/c.Ny/(c.MM-c.MM0))
    NRMSE = RMSE1 / np.var(Dp)

   

######################################################################################
     # Results
    c.RMSE1=RMSE1
    c.NRMSE=NRMSE

#####################################################################################
    if c.plot:
        plt.plot(Dp)
        plt.plot(Yp)
        plt.show()


if __name__ == "__main__":
    
    Config = Config2
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
