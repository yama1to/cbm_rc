# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc　時系列生成タスク　
cbmrc6e.pyを改変
Configクラスによるパラメータ設定

０から予測しさらにその予測を新たな入力とする
使わない
使わない
使わない
使わない
使わない
使わない
使わない
使わない
使わない
使わない
使わない
使わない
使わない
使わない
使わない

"""

import argparse
from matplotlib.colors import Normalize
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import time
from explorer import common
from generate_data_sequence_narma import *
from generate_matrix import *
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
        self.MM=1000 # サイクル数
        self.MM0 = 0 #

        self.Nu = 1   #size of input
        self.Nh = 500 #size of dynamical reservior
        self.Ny = 1   #size of output

        self.Temp=1.0
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.1
        self.alpha_r = 0.75
        self.alpha_b = 0.
        self.alpha_s = 2

        self.beta_i = 0.2
        self.beta_r = 0.1
        self.beta_b = 0.1

        self.lambda0 = 0.0001

        # Results
        self.NMSE=None
        self.RMSE=None
        self.NRMSE=None
        self.DC = None
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
    global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y, Us, Ds,Rs,hp,hx
    
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
    for n in tqdm(range(c.NN * c.MM), leave=True):
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
            if mode==0:
                print(hp)
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
    global Yp
    #print(Wo.shape,Hp.shape)
    Yp = Wo @ Hp.T

def test_network():
    run_network(0)


def predict():
    global Yp,Hp,hsign,Up,hx,hs,hc,hp,yp,rs,rs_prev
    
    hsign = np.zeros(c.Nh)
    hx = np.zeros(c.Nh)
    #hx = np.random.uniform(0,1,c.Nh) # [0,1]の連続値
    #hx = Hp[-1]
    hs = np.zeros(c.Nh) # {0,1}の２値
    Hp = np.zeros((c.MM, c.Nh))

    hc = np.zeros(c.Nh) # ref.clockに対する位相差を求めるためのカウント
    #hp = np.zeros(c.Nh) # [-1,1]の連続値

    Yp = np.zeros((c.MM, c.Ny))
    yp = np.zeros(c.Ny)
    rs = 1
    rs_prev = 0


    ########################
    yp = Wo@hp
    Up = Yp[-10]
    Yp[:10] = Yp[-10:]

    for i in tqdm(range(10,c.MM)):
            
        
        for n in range(c.NN):
            theta = np.mod(n/c.NN,1) # (0,1)
            rs_prev = rs

            rs = p2s(theta,0)# 参照クロック
            us = p2s(theta,Up) # エンコードされた入力

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
                yp = Wo@hp
                Up = yp
                Yp[i] = yp
                Hp[i] = hp

    return 0




def plot0():
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
    ax.set_title("Yp,dp")
    ax.plot(Yp)
    ax.plot(Dp)

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.set_title("Yp")
    ax.plot(Yp)

    plt.show()

def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,Yp
    global init_val,U2
    t_start=time.time()
    c.Nh = int(c.Nh)
    np.random.seed(int(c.seed))

    generate_weight_matrix()

    ### generate data
    if c.dataset==1:
        #MM1 = 1000
        #MM2 = 2200#2200]
        MM1 = 900
        MM2 = 100
        U,D  = generate_narma(N=MM1+MM2)
        U1,D1,U2,D2 = U[:MM1],D[:MM1],U[MM1:],D[MM1:]
        #_, D2  = generate_narma(N=MM2)

        #print(U1.shape)

    ### training
    #print("training...")
    c.MM=MM1
    
    Dp = D1
    Up = np.tanh(U1)
    train_network()
    Yp = Yp.T

    if 0:
        corr = np.corrcoef(np.vstack((Dp.T[0], Yp.T[0])))
        DC = corr[0,1]**2
        print("DC: %s" % str(DC))

        if c.plot:
            plt.plot(Dp,label="d")
            plt.plot(Yp,label="y")
            plt.legend()
            plt.show()
    ### test
    #print("test...")
    #print(Wo)

    #predict

    c.MM = MM2

    Dp  = D2
    
    predict()

    #Dp = fy(D2[200:])
    #Yp = Yp[200:]
    #Yp = Yp[10:]
    #Dp = Dp[10:]

    corr = np.corrcoef(np.vstack((Dp.T[0], Yp.T[0])))
    DC = corr[0,1]**2
    print("DC: %s" % str(DC))

    ### evaluation ######################################
    def mse(Yp,Dp):
        error = (Yp-Dp)**2
        ave = np.mean(error)
        return ave

    VAR = np.var(Dp)
    MEAN = np.mean(Dp)
    normalize = VAR


    MSE = mse(Yp,Dp)

    NMSE = MSE/normalize#mse(Dp,np.zeros(Dp.shape))
    #print("NMSE: ",NMSE)

    RMSE = np.sqrt(MSE)
    #print("RMSE: ",RMSE)

    NRMSE = RMSE/normalize
    #print("NRMSE: ",NRMSE)
    ###########################################################################
    c.NMSE = NMSE
    c.RMSE = RMSE
    c.NRMSE = NRMSE
    c.DC = DC
    c.cnt_overflow=cnt_overflow
    ###########################################################################
    #print("time: %.6f [sec]" % (time.time()-t_start))
    
    if c.plot: 
        plt.plot(Dp,label="d")
        plt.plot(Yp,label="y")
        plt.legend()
        plt.show()
        #plot0()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute()
    if a.config: common.save_config(c)
