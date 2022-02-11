# Copyright (c) 2018-2021 Katori lab. All Rights Reserved
"""
NOTE: cbm_rc 時系列生成タスク 
cbmrc6e.pyを改変
Configクラスによるパラメータ設定
"""

import argparse
from joblib import parallel_backend
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import copy
import time
from explorer import common
from generate_data_sequence import *
from generate_matrix import *
from tqdm import tqdm
from cbm_utils import *
class Config():
    def __init__(self):
        # columns, csv, id: データの管理のために必須の変数
        self.columns = None # 結果をCSVに保存する際のコラム
        self.csv = None # 結果を保存するファイル
        self.id  = None
        self.plot = 0 # 図の出力のオンオフ
        self.show = False # 図の表示（plt.show()）のオンオフ、explorerは実行時にこれをオフにする。
        self.savefig = False
        self.fig1 = "fig1.png" ### 画像ファイル名

        # config
        self.dataset=6
        self.seed:int=1 # 乱数生成のためのシード
        self.NN=2**8 # １サイクルあたりの時間ステップ
        self.MM=1200 # サイクル数
        self.MM0 = 200 #

        self.Nu = 1         #size of input
        self.Nh:int = 20   #815 #size of dynamical reservior
        self.Ny = 20        #size of output

        self.Temp=1
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.24
        self.alpha_r = 0.64
        self.alpha_s = 1.0
        self.beta_i = 0.36
        self.beta_r = 0.76

        # self.alpha_i2 = 0.24
        # self.alpha_r2 = 0.64
        # self.beta_i2 = 0.36
        # self.beta_r2 = 0.76
        

        # self.alpha_i3 = 0.24
        # self.alpha_r3 = 0.64
        # self.beta_i3 = 0.36
        # self.beta_r3 = 0.76

        self.alpha_b = 0.
        self.beta_b = 0.1

        self.alpha0 = 0#0.1
        self.alpha1 = 1#-5.8
        
        

        self.lambda0 = 0.

        self.delay = 20
        self.parallel = 5

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
def ring_weight():
    global Wr, Wb, Wo, Wi
    #taikaku = "zero"
    taikaku = "nonzero"
    Wr = np.zeros((c.Nh,c.Nh))
    for i in range(c.Nh-1):
        Wr[i,i+1] = 1
    
    # #print(Wr)
    # v = np.linalg.eigvals(Wr)
    # lambda_max = max(abs(v))
    # Wr = Wr/lambda_max*c.alpha_r
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
    
def small_world_weight():
    global Wr, Wb, Wo, Wi
    Wr = np.zeros((c.Nh,c.Nh))
    m = 0
    x = np.array([1,-1])
    np.random.shuffle(x)
    for i in range(c.Nh):
        Wr[i,i-2] = x[0]
        np.random.shuffle(x)
        Wr[i,i-1] = x[0]
        np.random.shuffle(x)
        rdm = np.random.uniform(0,1,1)
        if rdm > c.beta_r:
            Wr[i,int(np.random.randint(0,c.Nh-1,1))] = x[0]
            m+=1
    v = np.linalg.eigvals(Wr)
    lambda_max = max(abs(v))
    Wr = Wr/lambda_max*c.alpha_r
    return Wr

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

def fy(h):
    return np.tanh(h)

def fyi(h):
    return np.arctanh(h)

def p2s(theta,p):
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1)

def run_network(mode):
    global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y, Us, Ds,Rs
    Hp = np.zeros((c.MM, c.Nh*c.parallel))
    Hx = np.zeros((c.MM*c.NN, c.Nh))
    Hs = np.zeros((c.MM*c.NN, c.Nh))

    hsign = np.zeros((c.Nh,c.parallel))
    hx = np.zeros((c.Nh,c.parallel))
    hs = np.zeros((c.Nh,c.parallel))

    hs_prev = np.zeros(c.Nh)

    hc = np.zeros((c.Nh,c.parallel)) # ref.clockに対する位相差を求めるためのカウント
    hp = np.zeros((c.Nh,c.parallel)) # [-1,1]の連続値

    ht = np.zeros((c.Nh,c.parallel)) # {0,1}
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

    us = np.zeros((1,c.parallel))
    us_prev =np.zeros((1,c.parallel))

    group = np.zeros((c.Nh,c.parallel))
    for i in range(c.parallel):
        group[:,i] = i


    for n in tqdm(range(c.NN * c.MM)):
        
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        
        hs_prev = hs.copy()
        rs = p2s(theta,0)# 参照クロック
        us_prev = us.copy()
        for i in range(c.parallel-1):
            us[0,i+1] = us_prev[0,i]
        
        #or i in range(c.parallel):
        us[:,0] = p2s(theta,Up[m]) # エンコードされた入力

        ds = p2s(theta,Dp[m]) #
        ys = p2s(theta,yp)

        sum = np.zeros((c.Nh))
        
        # for i in range(c.parallel):
        #     #sum += c.alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
        #     sum += c.alpha_s*(hs[:,i]-rs)*ht[:,i] # ref.clockと同期させるための結合
        #     sum += Wi@(2*us[:,i]-1) # 外部入力
        #     #sum += us
        #     #sum += Wr@(2*hs-1) # リカレント結合
            
        #     sum += Wr@(2*p2s(theta,hp[:,i])-1) # リカレント結合
            
        #     hsign = 1 - 2*hs[:,i]
        #     hx[:,i] += hsign*(1.0+np.exp(hsign*sum/c.Temp))*c.dt

        sum = np.zeros((c.Nh,c.parallel))
        sum += c.alpha_s*(hs-rs)*ht# ref.clockと同期させるための結合
        sum += Wi@(2*us-1) # 外部入力
        #sum += us
        #sum += Wr@(2*hs-1) # リカレント結合
        
        sum += Wr@(2*p2s(theta,hp)-1) # リカレント結合
        
        hsign = 1 - 2*hs
        hx+= hsign*(1.0+np.exp(hsign*sum/c.Temp))*c.dt

        hs = np.heaviside(hx+hs-1,0)
        hx = np.fmin(np.fmax(hx,0),1)
        hc[(hs_prev == 1)& (hs==0)] = count

        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            hp_all = hp.reshape((c.Nh*c.parallel))

            hc = np.zeros((c.Nh,c.parallel)) #カウンタをリセット
            ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト

            yp = Wo@hp_all  
            # record    
            Hp[m]=hp_all
            Yp[m]=yp
            count = 0
            m += 1

        #境界条件
        if n == (c.NN * c.MM-1):
            hp = 2*hc/c.NN-1 # デコード、カウンタの値を連続値に変換
            hp_all = hp.reshape((c.Nh*c.parallel))

            yp = Wo@hp_all
            # record
            Hp[m]=hp_all
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

    ### generate data
    
    if c.dataset==6:
        T = c.MM
        #U,D = generate_white_noise(c.delay,T=T+200,)
        U,D = generate_white_noise(c.delay,T=T+200,dist="uniform")
        U=U[200:]
        D=D[200:]
    ### training
    #print("training...")
    max = np.max(np.max(abs(U)))
    if max>0.5:
        D /= max*2
        U /= max*2
    #Scale to (-1,1)
    Dp = D                # TARGET   #(MM,len(delay))   
    Up = U                # INPUT    #(MM,1)
    # plt.plot(U)
    # plt.tight_layout()
    # plt.savefig("memory-u.eps")
    #Up = D
    train_network()
    # print(U.shape,Hp.shape)
    # for i in range(20):
    #     corr = np.corrcoef(np.vstack((U[i:,0],Hp[i:,i])))
    #     corr = corr[0,1]
    #     print(corr)
    # plt.scatter(U[:,0],Hp[:,0])
    # plt.title("corr = %s"% str(corr))
    # plt.show()
    # exit()
    
    ### test
    
    #print("test...")
    # c.MM = c.MM - c.MM0
    # Dp = Dp[c.MM0:]                    # TARGET    #(MM,len(delay))
    # Up = Up[c.MM0:]                    # PRED      #(MM,len(delay))
    
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
    
   
######################################################################################
     # Results
    c.RMSE1=None
    c.RMSE2=None
    c.cnt_overflow=cnt_overflow

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
    print("MC =",c.MC)

#####################################################################################
    if c.plot:
        # fig=plt.figure(figsize=(12, 10))
        # ax = fig.add_subplot(2,1,1)
        # ax.cla()
        # ax.set_title("internal states")
        # ax.plot(Hx[50*256:100*256])
        # ax.set_xlabel("timestep")
        # ax = fig.add_subplot(2,1,2)
        # ax.cla()
        # ax.set_title("decoded internal states")
        # ax.plot(Hp[50:100])
        # ax.set_xlabel("time")
        # plt.show()
        #plot_delay(DC,4,Yp,Dp)
        plot_MC(DC,c.delay,MC)
        #plot1(Up,Us,Rs,Hx,Hp,Yp,Dp)
        #plot1()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    execute(c)
    if a.config: common.save_config(c)
