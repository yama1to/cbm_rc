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

from tqdm import tqdm 

class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 1#False # 図の出力のオンオフ
        self.show = 1#False # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = False
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=4
        self.seed:int=2 # 乱数生成のためのシード
        self.NN=256 # １サイクルあたりの時間ステップ
        self.MM=50 # サイクル数
        self.MM0 = 0 #

        self.Nu = 1   #size of input
        self.Nh:int = 100 #size of dynamical reservior
        self.Ny = 1   #size of output
    
        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.62
        self.alpha_r = 0.68
        self.alpha_b = 0.
        self.alpha_s = 1.49

        self.alpha0 = 0#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.47
        self.beta_r = 0.1
        self.beta_b = 0.1

        self.lambda0 = 0.0

        # Results
        self.RMSE1=None
        self.RMSE2=None
        self.cnt_overflow=None
        self.BER = None

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr")
    Wb = generate_random_matrix(c.Nh,c.Ny,c.alpha_b,c.beta_b,distribution="one",normalization="none")
    Wi = generate_random_matrix(c.Nh,c.Nu,c.alpha_i,c.beta_i,distribution="one",normalization="none")
    Wo = np.zeros(c.Nh * c.Ny).reshape((c.Ny, c.Nh))

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
    any_hs_change = True
    count =0
    m = 0
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

        hc[(hs_prev == 1)& (hs==0)] = count
        
        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            hc = np.zeros(c.Nh) #カウンタをリセット
            ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト
            yp = Wo@hp
            # record    
            Hp[m]=hp
            Yp[m]=yp
            count = 0
            m += 1

        #境界条件
        if n == (c.NN * c.MM-1):
            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            yp = Wo@hp
            # record
            Hp[m]=hp
            Yp[m]=yp

        count += 1
        any_hs_change = np.any(hs!=hs_prev)

        if c.plot:
        # record
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
    Nr=4
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    #ax.set_title("input")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    #ax.set_title("decoded reservoir states")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    #ax.set_title("predictive output")
    #ax.plot(train_Y)
    ax.plot(train_Y_binary)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    #ax.set_title("desired output")
    ax.plot(Dp)
    plt.savefig("./eps-fig/xor.eps")
def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2
    global train_Y_binary,train_Y
    c.NN = int(c.NN)
    np.random.seed(seed = int(c.seed))
    generate_weight_matrix()

    ### generate data
    if c.dataset==4:
        MM1=c.MM
        MM2 = c.MM
        U,D = generate_xor(MM1+MM2 +1)
        U1 = U[:MM1]
        D1 = D[:MM1]
        plt.plot(U,marker="o")
        plt.tight_layout()
        plt.savefig("xor-u.eps")
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
    c.RMSE1=None
    c.RMSE2=None
    c.cnt_overflow=cnt_overflow
    c.BER = BER
    #####################################################################################

    if c.plot: plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute()
    if a.config: common.save_config(c)