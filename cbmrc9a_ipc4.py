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
from generate_data_sequence_ipc2 import *
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
        self.NN=2**8 # １サイクルあたりの時間ステップ
        self.MM=2200 # サイクル数
        self.MM0 = 200 #

        self.Nu = 1   #size of input
        self.Nh = 100 #size of dynamical reservior
        self.Ny = 200   #size of output

        self.Temp=1.0
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.34
        self.alpha_r = 0.9
        self.alpha_b = 0.
        self.alpha_s = 0.8

        self.beta_i = 0.4
        self.beta_r = 0.25
        self.beta_b = 0.1

        self.lambda0 = 0.

        self.delay = 20
        self.degree = 10
        self.set = 0    #0,1,2,3
        # Results
        self.MC = None
        self.CAPACITY = None 
        self.cnt_overflow=None
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
def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    #Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr")
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
        #sum += Wr@(2*hs-1) # リカレント結合
        sum += Wr@(2*p2s(theta,hp)-1) # リカレント結合

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
        #print(G.shape,M.shape)
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
    fig=plt.figure(figsize=(10, 6))
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
    #plt.savefig(c.fig1)


def execute(c):
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global Yp,Dp,CAPACITY,sumOfCapacity, name ,dist 
    t_start=time.time()
    #if c.seed>=0:
    c.Nh = int(c.Nh)
    c.seed = int(c.seed)
    #c.Ny = c.delay
    np.random.seed(c.seed)    
    

    ### generate data
    name_list = ["Legendre","Hermite","Chebyshev","Laguerre"]
    dist_list = ["uniform","normal","arcsine","exponential"]
    dist = dist_list[c.set]
    name = name_list[c.set]
    
    U,D = datasets(k=c.delay,n=1,T = c.MM,name=name,dist=dist,seed=c.seed,new=0)
    for degree in range(2,11):
        _,D2 = datasets(k=c.delay,n=degree,T = c.MM,name=name,dist=dist,seed=c.seed,new=0)
        D = np.hstack([D,D2])
    #print(U.shape,D.shape)
    # max = np.max(np.max(abs(D)))
    # D /= max*1.01
    # U /= max*1.01
    # plt.plot(D[:,0])
    #plt.plot(U)
    # plt.show()

    generate_weight_matrix()
    ### training
    #print("training...")
    
    Dp = D[:]                # TARGET   #(MM,len(delay))   
    Up = U[:]                # INPUT    #(MM,1)

    train_network()
    #print("...end") 
    
    ### test
    #print("test...")
    Dp = D[:]                    # TARGET   #(MM,len(delay))   
    Up = U[:]                    # INPUT    #(MM,1)
    test_network()                  #OUTPUT = Yp

    ### evaluation

    Yp = Yp[c.MM0:]
    Dp = Dp[c.MM0:]
    # plt.plot(Yp)
    # plt.plot(Dp)
    # plt.show()
    MC = 0
    CAPACITY = []

   #print(Dp.shape,Yp.shape)
    for i in range(c.delay*10):
        r = np.corrcoef(Dp[c.delay:,i],Yp[c.delay:,i])[0,1]
        CAPACITY.append(r**2)
    MC = sum(CAPACITY)
    # ep = 1.7*10**(-4)
    # MC = np.heaviside(MC-ep,1)*MC
    
    SUM = np.sum((Yp-Dp)**2)
    #RMSE1 = np.sqrt(SUM/c.Ny/(c.MM-c.MM0-c.delay))

    #RMSE2 = 0
    print("-------------"+name+","+dist+",degree = "+str(c.degree)+"-------------")
    #print(CAPACITY)
    #print("RMSE=",RMSE1)
    #print("IPC=",MC)


######################################################################################
     # Results8

    c.MC = MC
    c.CAPACITY = CAPACITY
    c.cnt_overflow = cnt_overflow
#####################################################################################
    # plt.plot(Yp)
    # plt.show()
    # plt.plot(Dp)
    # plt.show()
    # plt.plot(Up)
    # plt.show()
    if c.plot:
        
        for i in range(1):
            c.plot = 1
            c.degree = i
            for i in range(10):
                plt.plot(c.CAPACITY[i*(20):(i+1)*20],label="degree = "+str(1+i))

                # plt.bar([c.alpha_i],[c.CAPACITY],bottom=prev,width=0.1,label=str(i+1))
                # prev+=c.CAPACITY
                # c.per.append([[c.alpha_i],[c.CAPACITY]]
        
        plt.ylabel("Capacity")
        plt.xlabel("delay")
        plt.ylim([-0.1,1.1])
        plt.xlim([-0.1,20.1])
        plt.title("cbm::"+c.name+"::"+c.dist)
        plt.legend()
        #plt.show()
        t = common.string_now()
        na = "./eps-fig/%s-ipc4_cbm_fixed_in_tar_%s.eps" % (t,str(c.set))
        plt.savefig(na)
        plt.clf()
    # if c.plot: plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    degree  = c.degree
    name_list = ["Legendre","Hermite","Chebyshev","Laguerre"]
    dist_list = ["uniform","normal","arcsine","exponential"]
    for c.set in range(4):
        c.dist = dist_list[c.set]
        c.name = name_list[c.set]
        execute(c)
    
     
    if a.config: common.save_config(c)
    # if a.config: c=common.load_config(c)
    # execute()
    # if a.config: common.save_config(c)
