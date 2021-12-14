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
from generate_data_sequence_ipc2 import *
from generate_matrix import *

class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = True # 図の出力のオンオフ
        self.show = 1 # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = False
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=6
        self.seed:int=2 # 乱数生成のためのシード
        self.MM=1000 # サイクル数
        self.MM0 = 100 #

        self.Nu = 1   #size of input
        self.Nh:int = 100#815 #size of dynamical reservior
        self.Ny = 1   #size of output


        #sigma_np = -5
        self.alpha_i = 0.1
        self.alpha_r = 0.9
        self.alpha_b = 0.

        self.alpha0 = 1#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.1
        self.beta_r = 0.9
        self.beta_b = 0.1

        self.lambda0 = 0

        self.delay = 20
        self.degree = 10
        self.set = 0    #0,1,2,3
        # Results
        self.CAPACITY = None
        self.MC = None 



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
    #x = np.random.uniform(-1, 1, c.Nh)
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

# def plot2():
#     plt.plot(c.CAPACITY)
#     plt.xlabel("delay k")
#     plt.ylabel("capacity")
#     plt.title("esn: units=%d,data = %d,trainsient=%d, sum of capacity=%.2lf \n poly = %s,dist = %s, degree = %d " \
#         % (c.Nh,c.MM,c.MM0,sumOfCAPACITY,name,dist,c.n_k[0,0]))
#     plt.ylim([-0.1,1.1])
#     #plt.show()
#     file_name = common.string_now()+"_"+name+"_"+dist+"_esn"
#     plt.savefig("./ipc_fig_dir/"+file_name)



def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global Yp,Dp,CAPACITY,name ,dist 
    t_start=time.time()
    #if c.seed>=0:
    c.Nh = int(c.Nh)
    c.seed = int(c.seed)
    c.Ny = c.delay
    np.random.seed(c.seed)    
    

    ### generate data
    name_list = ["Legendre","Hermite","Chebyshev","Laguerre"]
    dist_list = ["uniform","normal","arcsine","exponential"]
    dist = dist_list[c.set]
    name = name_list[c.set]
    
    U,D = datasets(k=c.delay,n=c.degree,T = c.MM,name=name,dist=dist,seed=c.seed,new=1)

    max = np.max(np.max(abs(D)))
    D /= max*1.01
    # plt.plot(D)
    # plt.plot(U)
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
    MC = 0
    CAPACITY = []
    for i in range(c.delay):
        r = np.corrcoef(Dp[c.delay:,i],Yp[c.delay:,i])[0,1]
        print(r**2)
        CAPACITY.append(r**2)
    MC = sum(CAPACITY)
    ep = 1.7*10**(-4)
    MC = np.heaviside(MC-ep,1)*MC
    
    SUM = np.sum((Yp-Dp)**2)
    RMSE1 = np.sqrt(SUM/c.Ny/(c.MM-c.MM0-c.delay))

    #RMSE2 = 0
    print("-------------"+name+","+dist+",degree = "+str(c.degree)+"-------------")
    print("RMSE=",RMSE1)
    print("IPC=",MC)

######################################################################################
     # Results8

    c.MC = MC
    c.CAPACITY = CAPACITY
#####################################################################################
    # plt.plot(Yp)
    # plt.show()
    # plt.plot(Dp)
    # plt.show()
    # plt.plot(Up)
    # plt.show()
    if c.plot: plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    degree  = c.degree

    plt.title("legendre")
    for i in range(1,degree+1):
        c.plot = 0
        c.degree = i
        execute(c)
        plt.plot(c.CAPACITY,label="degree = "+str(i))

            # plt.bar([c.alpha_i],[c.CAPACITY],bottom=prev,width=0.1,label=str(i+1))
            # prev+=c.CAPACITY
            # c.per.append([[c.alpha_i],[c.CAPACITY]]
    plt.ylabel("Capacity")
    plt.xlabel("delay")
    plt.ylim([-0.1,1.1])
    plt.xlim([-0.1,20.1])
    plt.legend()
    plt.show()
     
    if a.config: common.save_config(c)
    # if a.config: c=common.load_config(c)
    # execute()
    # if a.config: common.save_config(c)
