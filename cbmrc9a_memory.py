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
        self.seed:int=3 # 乱数生成のためのシード
        self.NN=256 # １サイクルあたりの時間ステップ
        self.MM=500 # サイクル数
        self.MM0 = 200 #

        self.Nu = 1   #size of input
        self.Nh:int = 20 #size of dynamical reservior
        self.Ny = 20   #size of output

        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.221
        self.alpha_r = 0.261
        self.alpha_b = 0.
        self.alpha_s = 0.033

        self.alpha0 = 0#0.1
        self.alpha1 = 0#-5.8

        self.beta_i = 0.1
        self.beta_r = 0.1
        self.beta_b = 0.1

        self.lambda0 = 0.2

        # Results
        self.RMSE1=None
        self.RMSE2=None
        self.MC = None 
        self.cnt_overflow=None
        #self.BER = None
        
        #self.DC = None 


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
    #hx = np.zeros(Nh)
    hx = np.random.uniform(0,1,c.Nh) # [0,1]の連続値
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
    for n in range(c.NN * c.MM):
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)# 参照クロック
        us = p2s(theta,Up[m]) # エンコードされた入力
        ds = p2s(theta,Dp[m]) #
        ys = p2s(theta,yp)

        sum = np.zeros(c.Nh)
        sum += c.alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
        #sum += alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
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
            #ht = 2*hs-1 リファレンスクロック同期用ラッチ動作をコメントアウト
            yp = fy(Wo@hp)
            # record
            Hp[m]=hp
            Yp[m]=yp
            m+=1

        any_hs_change = np.any(hs!=hs_prev)

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
    invD = fyi(Dp)
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
    #ax.plot(y)

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.plot(DC)
    ax.set_ylabel("determinant coefficient")
    ax.set_xlabel("Delay k")
    ax.set_title('MC ~ %3.2lf' % MC, x=0.8, y=0.7)

    plt.show()
    plt.savefig(c.fig1)

def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,Yp
    global RMSE1,RMSE2
    global train_Y_binary,MC,DC

########################################################################################\
    global dataset,seed,NN,MM,MM0,Nu,Nh,Ny,Temp,dt
    global alpha_i,alpha_b,alpha_r,alpha_s,alpha0,alpha1
    global beta_i,beta_r,beta_b,lambda0


    dataset=c.dataset
    
    seed=int(c.seed) # 乱数生成のためのシード
    NN=c.NN # １サイクルあたりの時間ステップ
    MM=c.MM # サイクル数
    MM0 = c.MM0 #
   

    Nu = c.Nu   #size of input
    Nh = c.Nh #size of dynamical reservior
    Ny = c.Ny   #size of output

    Temp=c.Temp
    dt=c.dt #0.01

    #sigma_np = -5
    alpha_i = c.alpha_i
    alpha_r = c.alpha_r
    alpha_b = c.alpha_b
    alpha_s = c.alpha_s

    alpha0 = c.alpha0#0.1
    alpha1 = c.alpha1#-5.8

    beta_i = c.beta_i
    beta_r = c.beta_r
    beta_b = c.beta_b

    lambda0 = c.lambda0


########################################################################################



    t_start=time.time()
    #if c.seed>=0:
    np.random.seed(int(c.seed))
    
    generate_weight_matrix()


    ### generate data
    
    if c.dataset==6:
        delay_s = 20
        delay = np.arange(delay_s)
        T = c.MM
        U,D = generate_white_noise(T=T,delay_s = delay_s)

        MM1 = c.MM
    
    """
    書籍付随コードでは入力500を300にしてからネットワークに入れている。
    これの意味がわからない。乱数系列に過渡応答もクソもないので

    多分本来は入力を500いれて
    過渡応答後(T>200)のレザバー状態収集行列Gを300にしてWoutの計算をするのではないか
    と考えている。
    """

    D1 = D
    U1 = U 

    ### training
    #print("training...")
    c.MM=MM1
    
    #Scale to (-1,1) in R
    Dp = np.tanh(D1)                # TARGET   #(MM,len(delay))   
    Up = np.tanh(U1)                # INPUT    #(MM,1)

    train_network()

    c.MM=MM1 - MM0
    Dp = Dp[MM0:]
    Up = Up[MM0:]
    ### test
    #print("test...")
    test_network()                  #OUTPUT = Yp

    
    DC = np.zeros((len(delay), 1))  # 決定係数
    MC = 0.0                        # 記憶容量

    #inv scale
    Dp = fyi(Dp)                    # TARGET    #(MM,len(delay))
    Yp = fyi(Yp)                    # PRED      #(MM,len(delay))
    
    """
    予測と目標から決定係数を求める。
    決定係数の積分が記憶容量

    """
    for k in range(len(delay)):
        corr = np.corrcoef(np.vstack((Dp.T[k, k:], Yp.T[k, k:])))   #相関係数
        DC[k] = corr[0, 1] ** 2                                     #決定係数 = 相関係数 **2

    MC = np.sum(DC)
   
######################################################################################
     # Results
    c.RMSE1=None
    c.RMSE2=None
    c.cnt_overflow=cnt_overflow
    #c.BER = None
    #c.DC = DC 
    c.MC = MC
    print(c.MC)#c.cnt_overflow)

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