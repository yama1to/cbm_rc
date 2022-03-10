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

import cupy as cp

class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 0 # 図の出力のオンオフ
        self.show = True # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = True
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=1
        self.seed:int=0 # 乱数生成のためのシード
        self.NN=256 # １サイクルあたりの時間ステップ
        self.MM=2000 # サイクル数
        self.MM0 = 200 #

        self.Nu = 1   #size of input
        self.Nh = 40 #size of dynamical reservior
        self.Ny = 1   #size of output

        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.35
        self.alpha_r = 0.3
        self.alpha_b = 0.
        self.alpha_s = 1

        self.beta_i = 0.28
        self.beta_r = 0.51
        self.beta_b = 0.1

        self.lambda0 = 0.0001
        self.delay = 9
        self.parallel = 5
        # Results
        self.RMSE   =   None
        self.NRMSE  =   None
        self.NMSE   =   None
        self.cnt_overflow   =   None
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
    global Wr, Wb, Wo, Wi,Wr2,Wr3,Wi2,Wi3,Wh
    Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr",diagnal=0)
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

def p2s_gpu(theta,p):
    return cp.where(cp.sin(cp.pi*(2*theta-p))>0,1,0)

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
    Up_prev =cp.zeros((1,c.parallel))

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
        us[:] = p2s_gpu(theta,Up_prev[:]) # エンコードされた入力

        # ds = p2s(theta,Dp[m]) #
        # ys = p2s(theta,yp)


        #sum = cp.zeros((c.Nh,c.parallel))
        sum = c.alpha_s*(hs-rs)*ht# ref.clockと同期させるための結合
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
            
            Up_prev[:,0] = cp.asarray(Up[m])

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
        E = cp.identity(c.Nh*c.parallel)
        TMP1 = cp.linalg.inv(M.T@M + c.lambda0 * E)
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

    MM1 = 1200
    MM2 = 2200
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
    #c.cnt_overflow=cnt_overflow/(c.MM-2)
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
