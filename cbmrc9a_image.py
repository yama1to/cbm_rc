# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc　時系列生成タスク　
cbmrc6e.pyを改変
Configクラスによるパラメータ設定
"""

import argparse
import numpy as np
from numpy.lib.npyio import load
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import time
from explorer import common
from generate_data_sequence_image import *
from generate_matrix import *

import gc 
from tqdm import tqdm


class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = True # 図の出力のオンオフ
        self.show = True # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = True
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=1
        self.seed:int=0 # 乱数生成のためのシード
        self.NN=256 # １サイクルあたりの時間ステップ
        self.MM=500 # サイクル数
        self.MM0 = 0 #

        self.Nu = 28**2   #size of input
        self.Nh = 1000 #size of dynamical reservior
        self.Ny = 10   #size of output

        self.Temp=1.0
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.5
        self.alpha_r = 0.3
        self.alpha_b = 0.
        self.alpha_s = 0.6

        self.beta_i = 0.6
        self.beta_r = 0.1
        self.beta_b = 0.1

        self.lambda0 = 0.

        # Results
        self.RMSE1=None
        self.RMSE2=None
        self.WER = None
        self.cnt_overflow=None

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr")
    Wb = generate_random_matrix(c.Nh,c.Ny,c.alpha_b,c.beta_b,distribution="one",normalization="none")
    Wi = generate_random_matrix(c.Nh,c.Nu,c.alpha_i,c.beta_i,distribution="one",normalization="none")
    Wo = np.zeros(c.Nh * c.Ny).reshape(c.Ny, c.Nh)

def fy(h):
    return np.tanh(h)

def fyi(h):
    return np.arctanh(h)

def p2s(theta,p):
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1)

def run_network(mode):
    global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y, Us, Ds,Rs
    Hp = np.zeros((c.MM, c.Nh))
    Hx = np.zeros((c.MM*c.NN, c.Nh))
    Hs = np.zeros((c.MM*c.NN, c.Nh))
    hsign = np.zeros(c.Nh)
    hx = np.zeros(c.Nh)
    #hx = np.random.uniform(0,1,c.Nh) # [0,1]の連続値
    hs = np.zeros(c.Nh) # {0,1}の２値
    hs_prev = np.zeros(c.Nh)
    hc = np.zeros(c.Nh) # ref.clockに対する位相差を求めるためのカウント
    hp = np.zeros(c.Nh) # [-1,1]の連続値
    ht = np.zeros(c.Nh) # {0,1}

    Yp = np.zeros((c.MM, c.Ny))
    Yx = np.zeros((c.MM*c.NN, c.Ny))
    Ys = np.zeros((c.MM*c.NN, c.Ny))
    #ysign = np.zeros(Ny)
    yp = np.zeros(c.Ny)
    yx = np.zeros(c.Ny)
    ys = np.zeros(c.Ny)
    #yc = np.zeros(Ny)

    Us = np.zeros((c.MM*c.NN, c.Nu))
    Ds = np.zeros((c.MM*c.NN, c.Ny))
    Rs = np.zeros((c.MM*c.NN, 1))

    rs = 1
    rs_prev = 0
    any_hs_change = True
    m=0
    for n in tqdm(range(c.NN * c.MM)):
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)# 参照クロック
        us = p2s(theta,Up[m]) # エンコードされた入力
        ds = p2s(theta,Dp[m]) #
        ys = p2s(theta,yp)

        sum = np.zeros(c.Nh)
        #sum += c.alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
        sum += c.alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
        sum += Wi@(2*us-1) # 外部入力
        sum += Wr@(2*hs-1) # リカレント結合

        #if mode == 0:
        #    sum += Wb@ys
        #if mode == 1:  # teacher forcing
        #    sum += Wb@ds

        hsign = 1 - 2*hs
        hx = hx + hsign*(1.0+np.exp(hsign*sum/c.Temp))*c.dt
        hs = np.heaviside(hx+hs-1,0)
        hx = np.fmin(np.fmax(hx,0),1)

        if rs==1:
            hc+=hs # デコードのためのカウンタ、ref.clockとhsのANDでカウントアップ

        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            hc = np.zeros(c.Nh) #カウンタをリセット
            ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト
            yp = Wo@hp
            # record
            Hp[m]=hp
            Yp[m]=yp
            m+=1

        any_hs_change = np.any(hs!=hs_prev)

        # record
        if c.plot:
            Rs[n]=rs
            Hx[n]=hx
            Hs[n]=hs
            Yx[n]=yx
            Ys[n]=ys
            Us[n]=us
            Ds[n]=ds

    # オーバーフローを検出する。
    global cnt_overflow
    cnt_overflow = 0
    for m in range(2,c.MM-1):
        tmp = np.sum( np.heaviside( np.fabs(Hp[m+1]-Hp[m]) - 0.6 ,0))
        cnt_overflow += tmp
        #print(tmp)

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
    run_network(0)

def plot1():
    fig=plt.figure(figsize=(20, 12))
    Nr=6
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("Up")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("Us")
    ax.plot(Us)
    ax.plot(Rs,"r:")
    #ax.plot(R2s,"b:")

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Hx")
    ax.plot(Hx)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    ax.set_title("Hp")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,5)
    ax.cla()
    ax.set_title("Yp")
    ax.plot(Yp)

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.set_title("Dp")
    ax.plot(Dp)

    plt.show()
    plt.savefig(c.fig1)

def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2
    t_start=time.time()
    #if c.seed>=0:
    np.random.seed(int(c.seed))
    #np.random.seed(c.seed)

    generate_weight_matrix()

    ### generate data
    U1,U2,D1,D2 = load_datasets()


    MM1 = U1.shape[0]
    MM2 = U2.shape[0]
    D1 = D1.reshape((MM1,10))
    D2 = D2.reshape((MM2,10))

    ### training
    #print("training...")
    
    #Scale to (-1,1)
    c.MM = MM1
    Dp = D1                # TARGET   #(MM,len(delay))   
    Up = U1               # INPUT    #(MM,1)

    if not c.plot: 
        del D1,U1
        gc.collect()

    train_network()
    #print("...end") 
    
    ### test
    #print("test...")
    c.MM = MM2
    Dp = D2                # TARGET   #(MM,len(delay))   
    Up = U2                # INPUT    #(MM,1)
    test_network()                  #OUTPUT = Yp
    if not c.plot: 
        del D2,U2
        gc.collect()
  

    for i in range(c.MM):
        idx = np.argmax(Yp[i])
        #print(idx)
        Yp[i] = np.zeros(10)
        Yp[i,idx] = 1
    
    ### evaluation
    sum=0
    SUM = np.sum(abs(Yp-Dp)/2)
    #print(Yp-Dp)
    SUM=np.sum(SUM)
    WER = SUM/c.MM
    print("学習回数: %d , テスト回数: %d , 誤り率: %.2lf," % (MM1,MM2,WER))

    #print(WSR)
######################################################################################
     # Results
    #c.RMSE1=RMSE1
    #c.RMSE2=RMSE2
    c.WSR = 1-WER
    c.WER = WER

    if c.plot: plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute()
    if a.config: common.save_config(c)
