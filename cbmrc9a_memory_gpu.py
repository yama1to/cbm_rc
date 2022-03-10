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
import os
import gc
from generate_data_sequence import *
from generate_matrix import *
from explorer import common

import cupy as cp 

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
        self.NN=256 # １サイクルあたりの時間ステップ
        self.MM=2000 # サイクル数
        self.MM0 = 100 #

        self.Nu = 1         #size of input
        self.Nh:int = 100   #815 #size of dynamical reservior
        self.Ny = 20        #size of output

        self.Temp=5
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 1
        self.alpha_r = 0.7
        self.alpha_b = 0.
        self.alpha_s = 2

        self.alpha0 = 0#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.9
        self.beta_r = 0.1
        self.beta_b = 0.1

        self.lambda0 = 0.
        self.delay = 20
        
        #self.dist = "normal"
        self.dist = "uniform"
        self.ave = 0
        self.std = 0.1

        # ResultsX
        self.RMSE1=None
        self.RMSE2=None
        self.MC = None
        self.MC1 = None 
        self.MC2 = None
        self.MC3 = None
        self.MC4 = None
        self.cnt_overflow=None
        #self.BER = None
        
        #self.DC = None 
def generate_weight_matrix():
    global Wr, Wb, Wo, Wi,Wr2,Wr3,Wi2,Wi3,Wh
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr",diagnal=1)
    # Wr2 = generate_random_matrix(c.Nh,c.Nh,c.alpha_r2,c.beta_r2,distribution="one",normalization="sr",diagnal=1)
    # Wr3 = generate_random_matrix(c.Nh,c.Nh,c.alpha_r3,c.beta_r3,distribution="one",normalization="sr",diagnal=1)
    #Wr = bm_weight()
    #Wr = ring_weight()
    #Wr = small_world_weight()
    Wb = generate_random_matrix(c.Nh,c.Ny,c.alpha_b,c.beta_b,distribution="one",normalization="none")
    Wi = generate_random_matrix(c.Nh,c.Nu,c.alpha_i,c.beta_i,distribution="one",normalization="none")
    
    # Wi2 = generate_random_matrix(c.Nh,c.Nu,c.alpha_i2,c.beta_i2,distribution="one",normalization="none")
    # Wi3 = generate_random_matrix(c.Nh,c.Nu,c.alpha_i3,c.beta_i3,distribution="one",normalization="none")

    #Wh = generate_random_matrix(c.Nh,c.Nh*3,c.alpha_i3,c.beta_i3,distribution="one",normalization="none")
    Wo = np.zeros(c.Nh*c.parallel * c.Ny).reshape((c.Ny, c.Nh*c.parallel))
    Wi = cp.asarray(Wi)
    Wr = cp.asarray(Wr)
    Wb = cp.asarray(Wb)
    Wo = cp.asarray(Wo)

def fy(h):
    return np.tanh(h)

def fyi(h):
    return np.arctanh(h)

def p2s(theta,p):
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1)

def run_network(mode):
    global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y, Us, Ds,Rs
    Hp = cp.zeros((c.MM, c.Nh*c.parallel))
    Hx = np.zeros((c.MM*c.NN, c.Nh))
    Hs = np.zeros((c.MM*c.NN, c.Nh))


    Yp = np.zeros((c.MM, c.Ny))
    Yx = np.zeros((c.MM*c.NN, c.Ny))
    Ys = np.zeros((c.MM*c.NN, c.Ny))
    #ysign = np.zeros(Ny)
    yp = np.zeros(c.Ny)
    yx = np.zeros(c.Ny)
    ys = np.zeros(c.Ny)
    Up_prev =np.zeros((1,c.parallel))
    #yc = np.zeros(Ny)

    Us = np.zeros((c.MM*c.NN, c.Nu))
    Ds = np.zeros((c.MM*c.NN, c.Ny))
    Rs = np.zeros((c.MM*c.NN, 1))

    hc = cp.zeros((c.Nh,c.parallel)) # ref.clockに対する位相差を求めるためのカウント
    hsign = cp.zeros((c.Nh,c.parallel))
    hx = cp.zeros((c.Nh,c.parallel))
    hs = cp.zeros((c.Nh,c.parallel))
    hs_prev = cp.zeros(c.Nh)
    hp = np.zeros((c.Nh,c.parallel)) # [-1,1]の連続値
    ht = cp.zeros((c.Nh,c.parallel)) # {0,1}
    us = cp.zeros((1,c.parallel))
    

    rs = 1
    any_hs_change = True
    count =0
    m = 0

    


    # group = np.zeros((c.Nh,c.parallel))
    # for i in range(c.parallel):
    #     group[:,i] = i


    for n in tqdm(range(c.NN * c.MM)):
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        
        hs_prev = hs.copy()
        rs = p2s(theta,0)# 参照クロック


        # for i in range(c.parallel-1):
        #     us[0,i+1] = us_prev[0,i]
        
        #or i in range(c.parallel):
        us[:] = cp.asarray(p2s(theta,Up_prev[:])) # エンコードされた入力

        # ds = p2s(theta,Dp[m]) #
        # ys = p2s(theta,yp)


        sum = cp.zeros((c.Nh,c.parallel))
        sum += c.alpha_s*(hs-rs)*ht# ref.clockと同期させるための結合
        #print(Wi.shape,us.shape)
        sum += Wi@(2*us-1) # 外部入力
        #sum += us
        sum += Wr@(2*hs-1) # リカレント結合

        #sum += Wr@(2*cp.asarray(p2s(theta,cp.asnumpy(hp)))-1) # リカレント結合
        
        hsign = 1 - 2*hs
        hx+= hsign*(1.0+np.exp(hsign*sum/c.Temp))*c.dt
        #hs = np.heaviside(hx+hs-1,0)
        hs = cp.where(hx>1,1,hs)
        hs = cp.where(hx<0,0,hs)
        hx = cp.fmin(cp.fmax(hx,0),1)
        #hc[(hs_prev == 1)& (hs==0)] = count
        hc = cp.where((hs_prev == 1)& (hs==0),count,hc)

        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            for i in reversed(range(c.parallel-1)):
                Up_prev[:,i+1] = Up_prev[0,i]
            
                Up_prev[:,0] = Up[m]

            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            hp_all = hp.reshape((c.Nh*c.parallel))

            hc = cp.zeros((c.Nh,c.parallel)) #カウンタをリセット
            ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト

            yp = Wo@hp_all
            # record    
            Hp[m]=hp_all
            Yp[m]=cp.asnumpy(yp)
            count = 0
            m += 1

        #境界条件
        if n == (c.NN * c.MM-1):
            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            hp_all = hp.reshape((c.Nh*c.parallel))

            yp = Wo@hp_all
            # record

            Hp[m]=hp_all
            Yp[m]=cp.asnumpy(yp)

        count += 1
        any_hs_change = np.any(hs!=hs_prev)

        if c.plot:
        # record
            Rs[n]=rs
            Hx[n]=hx
            Hs[n]=hs
            Yx[n]=yx
            Ys[n]=ys
            #Us[n]=us
            #Ds[n]=ds

    # # オーバーフローを検出する。
    # global cnt_overflow
    # cnt_overflow = 0
    # for m in range(2,c.MM-1):
    #     tmp = np.sum( np.heaviside( np.fabs(Hp[m+1]-Hp[m]) - 0.6 ,0))
    #     cnt_overflow += tmp
    #     #print(tmp)

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = Hp[c.MM0:, :]
    invD = Dp
    G = cp.asarray(invD[c.MM0:, :])

    #print("Hp\n",Hp)
    #print("M\n",M)

    ### Ridge regression
    if c.lambda0 == 0:
        Wo = cp.dot(G.T,cp.linalg.pinv(M).T)
        #print("a")
    else:
        E = cp.identity(c.Nh)
        TMP1 = cp.linalg.inv(M.T@M + c.lambda0 * E)
        WoT = TMP1@M.T@G
        Wo = WoT.T
    #print("WoT\n", WoT)

def test_network():
    run_network(0)

def plot1():
    fig=plt.figure(figsize=(16, 8))
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
    #ax.plot(y)

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.plot(DC)
    ax.set_ylabel("determinant coefficient")
    ax.set_xlabel("Delay k")
    ax.set_ylim([0,1])
    ax.set_xlim([0,c.delay])
    ax.set_title('MC ~ %3.2lf' % MC, x=0.8, y=0.7)

    plt.show()
    plt.savefig(c.fig1)

def plot_delay():
    fig=plt.figure(figsize=(16,16 ))
    Nr=20
    start = 0
    for i in range(20):
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
    plt.ylim([0,1.1])
    plt.xlim([0,c.delay])
    plt.title('MC ~ %3.2lf,Nh = %d' % (MC,c.Nh), x=0.8, y=0.7)

    if 1:
        fname = "./MC_fig_dir/MC:alphai={0},r={1},s={2},betai={3},r={4}.png".format(c.alpha_i,c.alpha_r,c.alpha_s,c.beta_i,c.beta_r)
        plt.savefig(fname)
    plt.show()

def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,Yp
    global RMSE1,RMSE2
    global train_Y_binary,MC,DC


    c.delay = int(c.delay)
    c.Ny = c.delay
    c.NN = int(c.NN)
    c.Nh = int(c.Nh)

    np.random.seed(seed = int(c.seed))    
    generate_weight_matrix()
    # for i in range(c.Nh):
    #     Wr[i,i] = 0

    ### generate data
    
    U,D = generate_white_noise(delay_s=c.delay,T=c.MM+200,dist = c.dist,ave = c.ave,std = c.std)
    U=U[200:]
    D=D[200:]
    ### training
    #print("training...")
    
    #Scale to (-1,1)
    Dp = D                # TARGET   #(MM,len(delay))   
    Up = U                # INPUT    #(MM,1)

    train_network()
    if not c.plot: 
        del D,U,Us,Rs
        gc.collect()
        

    
    
    ### test
    #print("test...")
    test_network()                  #OUTPUT = Yp


    
    DC = np.zeros((c.delay, 1))  # 決定係数
    MC = 0.0                        # 記憶容量

    #inv scale
    
    Dp = Dp[c.MM0:]                    # TARGET    #(MM,len(delay))
    Yp = Yp[c.MM0:]                    # PRED      #(MM,len(delay))
    #print(np.max(Dp),np.max(Yp))
    """
    予測と目標から決定係数を求める。
    決定係数の積分が記憶容量
    """
    for k in range(c.delay):
        corr = np.corrcoef(np.vstack((Dp.T[k, k:], Yp.T[k, k:])))   #相関係数
        DC[k] = corr[0, 1] ** 2                                     #決定係数 = 相関係数 **2

    MC = np.sum(DC)
    
    #plot_MC()
######################################################################################
     # Results
    c.RMSE1=None
    c.RMSE2=None
    c.cnt_overflow=cnt_overflow

    c.MC = MC
    #c.DC =DC

    # if c.delay >=5:
    #     MC1 = np.sum(DC[:5])
    #     c.MC1 = MC1

    # if c.delay >=10:
    #     MC2 = np.sum(DC[:10])
    #     c.MC2 = MC2

    # if c.delay >=20:
    #     MC3 = np.sum(DC[:20])
    #     c.MC3 = MC3

    # if c.delay >=50:
    #     MC4 = np.sum(DC[:50])
    #     c.MC4 = MC4
    print("MC =",c.MC)
    print("overflow =",c.cnt_overflow)
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
