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
from generate_data_sequence import *
from generate_matrix import *

from test_cbm_esn_memory import Config2

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
    x = np.random.uniform(-1, 1, c.Nh)
    #x = np.zeros(c.Nh)
    

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




def plot_delay():
    fig=plt.figure(figsize=(16,16 ))
    Nr=20
    if c.delay< 20: Nr = c.delay 
    
    for i in range(Nr):
            ax = fig.add_subplot(Nr,1,i+1)
            ax.cla()
            ax.set_title("Yp,Dp, delay = %s" % str(i))
            ax.plot(Yp.T[i,i:])
            ax.plot(Dp.T[i,i:])

    plt.show()


def plot_MC():
    plt.plot(DC)
    plt.ylabel("determinant coefficient")
    plt.xlabel("Delay k")
    plt.ylim([0,1])
    plt.xlim([0,c.delay])
    plt.title('MC ~ %3.2lf' % MC, x=0.8, y=0.7)
    plt.show()

def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,Yp
    global RMSE1,RMSE2
    global train_Y_binary,MC,DC


    c.delay = int(c.delay)
    c.Ny = c.delay
    c.Nh = int(c.Nh)

    np.random.seed(seed = int(c.seed))    
    generate_weight_matrix()

    ### generate data
    
    if c.dataset==6:
        T = c.MM
        U,D = generate_white_noise(c.delay,T=T,)

    ### training
    #print("training...")
    
    #Scale to (-1,1)
    Dp = D                # TARGET   #(MM,len(delay))   
    Up = U                # INPUT    #(MM,1)

    train_network()
    #print("...end") 
    
    ### test
    #print("test...")
    test_network()                  #OUTPUT = Yp

    #print("...end")
    Yp = Yp[c.MM0:]
    Dp = Dp[c.MM0:]
    DC = np.zeros((c.delay, 1))  # 決定係数
    MC = 0.0                        # 記憶容量


    #print(np.max(Dp),np.max(Yp))
    """
    予測と目標から決定係数を求める。
    決定係数の積分が記憶容量
    """ 


    for k in range(c.delay):
        corr = np.corrcoef(np.vstack((Dp.T[k, k:], Yp.T[k, k:])))   #相関係数
        DC[k] = corr[0, 1] ** 2                                     #決定係数 = 相関係数 **2


    MC = np.sum(DC)
    #print(MC)
   
    #print(MC,MC1,MC2,MC3,MC4)
######################################################################################
     # Results

    c.MC = MC

    print("MC =",c.MC)

#####################################################################################
    if c.plot:
        #plot_delay()
        plot_MC()
        #plot1()


if __name__ == "__main__":
    Config = Config2

    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
