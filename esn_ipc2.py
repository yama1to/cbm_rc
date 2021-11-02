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
from generate_data_sequence_ipc import *
from generate_matrix import *

class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = True # 図の出力のオンオフ
        self.show = False # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = False
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=6
        self.seed:int=2 # 乱数生成のためのシード
        self.MM=5000 # サイクル数
        self.MM0 = 4000 #

        self.Nu = 1   #size of input
        self.Nh:int = 100#815 #size of dynamical reservior
        self.Ny = 1   #size of output


        #sigma_np = -5
        self.alpha_i = 0.7
        self.alpha_r = 0.95
        self.alpha_b = 0.

        self.alpha0 = 1#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.8
        self.beta_r = 0.05
        self.beta_b = 0.1

        self.lambda0 = 0

        self.n_k    =   np.array([[2,1]])
        self.set = 3    #0,1,2,3
        #np.array([[1,1],[1,2]])

        # Results

        self.CAPACITY = None 
        self.sumOfCAPACITY = None 



def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="normal",normalization="sr")
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



def plot1():
    fig=plt.figure(figsize=(10, 6))
    Nr=3
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("U")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("X")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Y, Ytarget")
    ax.plot(Dp,label = "d",color = "b")
    ax.plot(Yp,label = "y",color = "r")
    plt.legend()
    plt.show()

def plot2():
    plt.plot(c.CAPACITY)
    plt.xlabel("delay k")
    plt.ylabel("capacity")
    plt.title("esn: units=%d,data = %d,trainsient=%d, sum of capacity=%.2lf \n poly = %s,dist = %s " \
        % (c.Nh,c.MM,c.MM0,sumOfCAPACITY,name,dist))
    plt.ylim([-0.1,1.1])
    plt.show()

def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global Yp,Dp,sumOfCAPACITY
    t_start=time.time()
    #if c.seed>=0:
    c.Nh = int(c.Nh)
    c.seed = int(c.seed)
    np.random.seed(c.seed)    

    c.Ny = 20

    delay = 20

    global name ,dist 

    name_list = ["Legendre","Hermite","Chebyshev","Laguerre"]
    dist_list = ["uniform","normal","arcsine","exponential"]

    dist = dist_list[c.set]
    name = name_list[c.set]

    for i in range(delay):
        n_k=np.array([[2,i]])
        if i==0:
            U,D = datasets(n_k=n_k,T = c.MM,name=name,dist=dist,seed=c.seed)
            #print(D.shape)
        else:
            _,d = datasets(n_k=n_k,T = c.MM,name=name,dist=dist,seed=c.seed)
            #print(d.shape)
            D = np.hstack((D,d))
    #plt.plot(U)
    # for i in range(D.shape[1]):
    #     plt.plot(D[i],label = str(i))
    # plt.legend()
    # plt.show()

    generate_weight_matrix()
    ### training
    #print("training...")
    
    Dp = D[:]                # TARGET   #(MM,len(delay))   
    Up = U[:]                # INPUT    #(MM,1)

    train_network()
    #print("...end") 
    
    ### test
    #print("test...")
    Dp = D[:]                    # TARGET   #(MM,len(delay))   
    Up = U[:]                    # INPUT    #(MM,1)
    test_network()                  #OUTPUT = Yp

    ### evaluation

    
    Yp = fy(Yp[c.MM0:])
    Dp = fy(Dp[c.MM0:])

    CAPACITY = np.zeros(delay)

    for i in range(delay):
        r = np.corrcoef(Dp[i:,i],Yp[i:,i])[0,1]
        CAPACITY[i] = r**2
    sumOfCAPACITY = np.sum(CAPACITY)

    SUM = np.sum((Yp-Dp)**2)
    RMSE1 = np.sqrt(SUM/c.Ny/Dp.shape[0])
    print("RMSE=",RMSE1)
    print("IPC=",CAPACITY)
    print("sum of IPC=",sumOfCAPACITY)


######################################################################################
     # Results
    c.CAPACITY = CAPACITY
    c.sumOfCAPACITY = sumOfCAPACITY
#####################################################################################

    if c.plot: 
        #plot1()
        plot2()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
