# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc　時系列生成タスク　
cbmrc6e.pyを改変
Configクラスによるパラメータ設定


入力から予測を出力する

"""

import argparse
import numpy as np
from numpy.core.fromnumeric import size
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import time
from explorer import common
from generate_data_sequence_narma import *
from generate_matrix import *
import gc
from tqdm import tqdm 


class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 1 # 図の出力のオンオフ
        self.show = True # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = True
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=1
        self.seed:int=0 # 乱数生成のためのシード
        self.NN=2**10 # １サイクルあたりの時間ステップ
        self.MM=2000 # サイクル数
        self.MM0 = 200 #

        self.Nu = 1   #size of input
        self.Nh = 300 #size of dynamical reservior
        self.Ny = 1   #size of output

        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.35
        self.alpha_r = 0.9
        self.alpha_b = 0.
        self.alpha_s = 0.47

        self.beta_i = 0.28
        self.beta_r = 0.51
        self.beta_b = 0.1

        self.lambda0 = 0.0001
        self.delay = 9
        # Results
        self.RMSE   =   None
        self.NRMSE  =   None
        self.NMSE   =   None
        self.cnt_overflow   =   None


def ring_weight():
    global Wr, Wb, Wo, Wi
    #taikaku = "zero"
    taikaku = "nonzero"
    Wr = np.zeros((c.Nh,c.Nh))
    for i in range(c.Nh-1):
        Wr[i,i+1] = 1
    Wr[-1,0]=1
    # #print(Wr)
    v = np.linalg.eigvals(Wr)
    lambda_max = max(abs(v))
    Wr = Wr/lambda_max*c.alpha_r
    return Wr

def bm_weight():
    global Wr, Wb, Wo, Wi
    #taikaku = "zero"
    taikaku = "nonzero"
    Wr = np.zeros((c.Nh,c.Nh))
    x = c.Nh**2
    if taikaku =="zero":
        x -= c.Nh
    nonzeros = int(x * c.beta_r)
    x = np.zeros((x))
    x[0:int(nonzeros / 2)] = 1
    x[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(x)
    m = 0
    
    for i in range(c.Nh):
        for j in range(i,c.Nh):
            if taikaku =="zero":
                if i!=j:
                    Wr[i,j] = x[m]
                    Wr[j,i] = x[m]
                    m += 1
            else:
                Wr[i,j] = x[m]
                Wr[j,i] = x[m]
                m += 1
    #print(Wr)
    v = np.linalg.eigvals(Wr)
    lambda_max = max(abs(v))
    Wr = Wr/lambda_max*c.alpha_r
    return Wr
def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr")
    #Wr  = bm_weight()
    Wr = ring_weight()
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
    
    hsign = np.zeros(c.Nh)
    hx = np.zeros(c.Nh)
    #hx = np.random.uniform(0,1,c.Nh) # [0,1]の連続値
    hs = np.zeros(c.Nh) # {0,1}の２値
    hs_prev = np.zeros(c.Nh)
    hc = np.zeros(c.Nh) # ref.clockに対する位相差を求めるためのカウント
    hp = np.zeros(c.Nh) # [-1,1]の連続値
    ht = np.zeros(c.Nh) # {0,1}


    #ysign = np.zeros(Ny)
    yp = np.zeros(c.Ny)
    yx = np.zeros(c.Ny)
    ys = np.zeros(c.Ny)
    Yp = np.zeros((c.MM, c.Ny))
    #yc = np.zeros(Ny)
    if c.plot:
        Us = np.zeros((c.MM*c.NN, c.Nu))
        Ds = np.zeros((c.MM*c.NN, c.Ny))
        Rs = np.zeros((c.MM*c.NN, 1))

        Hx = np.zeros((c.MM*c.NN, c.Nh))
        Hs = np.zeros((c.MM*c.NN, c.Nh))

        
        Yx = np.zeros((c.MM*c.NN, c.Ny))
        Ys = np.zeros((c.MM*c.NN, c.Ny))


    rs = 1
    any_hs_change = True
    count =0
    m = 0

    x = np.zeros((c.Nh))
    y = np.zeros((c.Nh))
    z = np.zeros((c.Nh))


    evolve = np.ones(c.Nh)

    for n in tqdm(range(c.NN*c.MM)):
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)# 参照クロック
        us = p2s(theta,Up[m]) # エンコードされた入力
        #us = p2s(theta,Wi@(2*Up[m]-1))
        ds = p2s(theta,Dp[m]) #
        ys = p2s(theta,yp)

        #print(us == Wi@(2*p2s(theta,Up[m])-1))
        # print("aaaaaaaaaaaaaaaaaaa")
        # print(us)
        # print(Wi@(2*p2s(theta,Up[m])-1))
        sum = np.zeros(c.Nh)
        #sum += c.alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
        sum += c.alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
        sum += Wi@(2*us-1) # 外部入力
        #sum += us
        #sum += Wr@(2*hs-1) # リカレント結合
        sum += Wr@(2*p2s(theta,hp)-1) # リカレント結合
        #if mode == 0:
        #    sum += Wb@ys
        #if mode == 1:  # teacher forcing
        #    sum += Wb@ds


        
        hsign = 1 - 2*hs
        hx[evolve==1] = hx[evolve==1] + (hsign*(1.0+np.exp(hsign*sum/c.Temp))*c.dt)[evolve==1]
        hs[evolve==1] = np.heaviside(hx[evolve==1]+hs[evolve==1]-1,0)
        hx[evolve==1] = np.fmin(np.fmax(hx[evolve==1],0),1)
        ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト
        
        y[(hs_prev == 0)& (hs==1)] = n
        
        # ２値状態 立ち下がり
        if np.sum((hs_prev == 1)& (hs==0))>0:
            id = (hs_prev == 1)& (hs==0)
            #duty ratio (z-y)/(z-x)
            z[id] = n
            hc[id] = 2*(z[id]-y[id])/(z[id]-x[id])-1
            x[id] = n
            #print(hp[id])
            #evolve[id] = 0
            
            #hc[id] = 1
            

        if rs_prev==0 and rs ==1:
            
            hp = hc # デコード、カウンタの値を連続値に変換
            hc = np.zeros(c.Nh) #カウンタをリセット
            #print(hp)
            evolve = np.ones(c.Nh)
            yp = Wo@hp

            # record    
            Hp[m]=hp
            Yp[m]=yp
            count = 0
            m += 1

        #境界条件
        if n == (c.NN * c.MM-1):
            hp = hc
            #hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
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
            #Us[n]=us
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
    invD =Dp
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
    ax.plot(Yp)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    #ax.set_title("desired output")
    ax.plot(Dp)
    plt.savefig("./eps-fig/narma.eps")
def plot2():
    fig=plt.figure(figsize=(20, 12))
    Nr=2
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("Yp,Dp")
    ax.plot(Yp,label = "prediction ")
    ax.plot(Dp, label = "Target")
    ax.legend()

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("error",size=10)
    ax.plot(abs(Yp-Dp))
    plt.xlabel("time")

    plt.show()

def calc(Yp,Dp):
    error = (Yp-Dp)**2
    NMSE = np.mean(error)/np.var(Dp)
    RMSE = np.sqrt(np.mean(error))
    NRMSE = RMSE/np.var(Dp)
    return RMSE,NRMSE,NMSE

def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM,Yp

    t_start=time.time()
    c.seed = int(c.seed)
    c.Nh = int(c.Nh)
    c.delay = int(c.delay)
    c.Ny = c.delay
    

    np.random.seed(c.seed)
    generate_weight_matrix()

    MM1 = 1000
    MM2 = 500
    U1,D1  = generate_narma(N=MM1,seed=0,delay=c.delay)
    U2,D2  = generate_narma(N=MM2,seed=1,delay=c.delay)
    # plt.plot(D1)
    # plt.tight_layout()
    # plt.savefig("narma-d.eps")
    #print((MM2-2))
    #print(np.var(D1))

    Dp = D1
    Up = U1 
    c.MM = MM1
    if not c.plot: 
        del D1,U1
        gc.collect()
    train_network()

    # RMSE1,NRMSE1 = calc(Yp,Dp)
    # print(RMSE1,NRMSE1)

    
        

    ### test
    #print("test...")
    c.MM = MM2
    Dp = D2
    Up = U2
    if not c.plot: 
        del U2,D2
        gc.collect()
    test_network()

    global Yp 
    Yp = Yp[c.MM0:]
    Dp = Dp[c.MM0:]

    RMSE,NRMSE,NMSE = calc(Yp,Dp)
    #print(1/np.var(Dp))
    print("RMSE:",RMSE,"NRMSE:",NRMSE,"NMSE:",NMSE)
    c.RMSE = RMSE
    c.NRMSE = NRMSE
    c.NMSE = NMSE
    c.cnt_overflow=cnt_overflow/(c.MM-2)
    #print("time: %.6f [sec]" % (time.time()-t_start))

    if c.plot:
        plot1()
        #plot2()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute()
    if a.config: common.save_config(c)
