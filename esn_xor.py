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
        self.dataset=4
        self.seed:int=2 # 乱数生成のためのシード
        self.MM=50 # サイクル数
        self.MM0 = 0 #

        self.Nu = 1   #size of input
        self.Nh:int = 20#815 #size of dynamical reservior
        self.Ny = 20   #size of output


        #sigma_np = -5
        self.alpha_i = 0.1
        self.alpha_r = 0.8
        self.alpha_b = 0.

        self.alpha0 = 1#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.9
        self.beta_r = 0.1
        self.beta_b = 0.1

        self.lambda0 = 0.0

        # Results
        self.BER = None 
        



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
    #x = np.random.uniform(-1, 1, c.Nh)
    
    for n in range(c.MM):
        
        u = Up[n, :]

        #Hp[n+1,:] = x + 1.0/tau * (-alpha0 * x + fx(Wi@u + Wr@x))
        next_x = (1 - c.alpha0) * x + c.alpha0*fy(Wi@u + Wr@x)
        #next_x = np.round(next_x,8)
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
    Nr=4
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("input")
    ax.plot(Up)


    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("reservoir states")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("predictive output")
    ax.plot(train_Y)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    ax.set_title("desired output")
    ax.plot(Dp)
    ax.plot(train_Y_binary)
    ax.plot()
    
    plt.show()
    plt.savefig(c.fig1)


def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2
    global train_Y_binary,train_Y
    np.random.seed(seed = int(c.seed))
    generate_weight_matrix()

    ### generate data
    if c.dataset==4:
        MM1= c.MM
        MM2 = c.MM
        U,D = generate_xor(MM1+MM2 +1)
        U1 = U[:MM1]
        D1 = D[:MM1]
        # plt.plot(U1)
        # plt.plot(D1)
        # plt.show()
    ### training
    #print("training...")
    c.MM = MM1
    Dp = D1
    Up = np.tanh(U1)
    train_network()                     #Up,Dpからネットワークを学習する

    ### test
    #print("test...")
    c.MM = MM2


    test_network()                      #output = Yp

    tau = 2
    # 評価（ビット誤り率, BER）
    train_Y_binary = np.zeros(MM1-tau)

    train_Y = Yp[tau:]        #(T-tau,1)
    Dp      = Dp[tau:]          #(T-tau,1)

    #閾値を0.5としてバイナリ変換する
    for n in range(MM1-tau):
        train_Y_binary[n] = np.heaviside(train_Y[n]-fy(0.5),0)
    
    BER = np.linalg.norm(train_Y_binary-Dp[:,0], 1)/(MM1-tau)


    print('BER ={:.3g}'.format(BER))
    ######################################################################################
     # Results

    c.BER = BER
    #####################################################################################

    if c.plot: plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
