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
        self.MM=300 # サイクル数
        self.MM0 = 0 #

        self.Nu = 1   #size of input
        self.Nh:int = 100#815 #size of dynamical reservior
        self.Ny = 20   #size of output


        #sigma_np = -5
        self.alpha_i = 1.
        self.alpha_r = 0.9
        self.alpha_b = 0.

        self.alpha0 = 0.1#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.9
        self.beta_r = 0.05
        self.beta_b = 0.1

        self.lambda0 = 0.0

        self.delay = 20


        # Results
        self.RMSE1=None
        self.RMSE2=None
        self.MC = None
        self.MC1 = None 
        self.MC2 = None
        self.MC3 = None
        self.MC4 = None



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
    

    for n in range(c.MM - 1):
        
        u = Up[n, :]

        #Hp[n+1,:] = x + 1.0/tau * (-alpha0 * x + fx(Wi@u + Wr@x))
        next_x = (1 - c.alpha0) * x + c.alpha0*fy(Wi@u + Wr@x)
        Hp[n+1,:] = next_x
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
    c.RMSE1=None
    c.RMSE2=None
    c.MC = MC

    if c.delay >=5:
        MC1 = np.sum(DC[:5])
        c.MC1 = MC1

    if c.delay >=10:
        MC2 = np.sum(DC[:10])
        c.MC2 = MC2

    if c.delay >=20:
        MC3 = np.sum(DC[:20])
        c.MC3 = MC3

    if c.delay >=50:
        MC4 = np.sum(DC[:50])
        c.MC4 = MC4
    
    
    
    
    #print("MC =",c.MC)

#####################################################################################
    if c.plot:
        #plot_delay()
        plot_MC()
        #plot1()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
