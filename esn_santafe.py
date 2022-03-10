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
from generate_data_sequence_santafe import *
from generate_matrix import *
import gc

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
        self.dataset=1
        self.seed:int=2 # 乱数生成のためのシード
        self.MM=500 # サイクル数
        self.MM0 = 0 #

        self.Nu = 1   #size of input
        self.Nh:int = 300#815 #size of dynamical reservior
        self.Ny = 1   #size of output


        #sigma_np = -5
        self.alpha_i = 0.1
        self.alpha_r = 0.95
        self.alpha_b = 0.

        self.alpha0 = 1#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.9
        self.beta_r = 0.05
        self.beta_b = 0.1

        self.lambda0 = 0.1
        self.delay = [1,2,3,4,5]

        # Results
        self.NRMSE=None
        self.RMSE=None



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



def plot1():
    fig=plt.figure(figsize=(20, 12))
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
    fig=plt.figure(figsize=(20, 12))
    Nr=2
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("Yp,Dp, delay = %s" % c.delay[0])
    ax.plot(list(range(train_num,train_num+test_num)),Yp,label = "prediction ")
    ax.plot(list(range(train_num,train_num+test_num)),Dp, label = "Target")
    ax.legend()

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("error")
    ax.plot(list(range(train_num,train_num+test_num)),abs(Yp-Dp))


    plt.show()
def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,Yp,delay,train_num,test_num
    global RMSE1,RMSE2

   
    
    c.Nh = int(c.Nh)
    ### generate data
    train_num = 1000
    test_num = 2000
    if c.dataset==1:
        #delay = 1,2,3,4,5
        c.delay = [1,2,3,4,5]
        U1,D1,U2,D2,normalize = generate_santafe(future = c.delay,train_num = train_num,test_num =test_num,)
    
    #print(D2[:,2]==D2[:,3])
    # plt.plot(U2,label="u")
    # for i in range(5):
    #     plt.plot(D2[:,i],label=i+1)
    # plt.legend()
    # plt.show()
    #print(normalize)
    c.Ny = int(len(c.delay))
    generate_weight_matrix()
    ### training
    #print("training...")
    c.MM= train_num

    Dp = D1
    Up = U1
    if c.plot:
        del U1,D1
        gc.collect()

    train_network()

    


    ### test
    #print("test...")
    c.MM= test_num

    Dp = D2
    Up = U2
    if c.plot:
        del U2,D2
        gc.collect()
    test_network()


    ### evaluation
    sum=0
    Yp = Yp*normalize
    Dp = Dp*normalize
 
    for j in range(c.MM):
        sum += (Yp[j] - Dp[j])**2
    MSE = sum/c.MM
    RMSE = np.sqrt(MSE)
    
    NRMSE = RMSE/np.var(Dp)#np.std(Dp)#np.var(Dp)
    c.RMSE = RMSE
    c.NRMSE2 = NRMSE

    NMSE =(MSE/np.var(Dp))
    print(NMSE)
    c.NMSE = np.sum(NMSE)/NMSE.size
    c.NRMSE = np.sum(NRMSE)/NRMSE.size

    #print(RRMSE1)
    #print("それぞれのdelayでのNRMSEを全部加算した場合のNRMSE: "+str(c.NRMSE))
    #print("time: %.6f [sec]" % (time.time()-t_start))

    if c.plot: 
        plot1()
        plot2()
        #print(RMSE)
        print("それぞれのdelayでのNMSEを全部加算した場合のNMSE: "+str(c.NMSE))



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
