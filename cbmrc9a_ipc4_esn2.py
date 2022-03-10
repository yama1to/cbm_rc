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
        self.plot = 0 # 図の出力のオンオフ
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
        self.Ny = 20   #size of output

        self.Temp=1.0
        self.dt=1.0/self.NN #0.01

        #sigma_np = -5
        self.alpha_i = 0.11
        self.alpha_r = 0.89
        self.alpha_b = 0.
        self.alpha_s = 1.58
        self.alpha0 = 0

        self.beta_i = 0.39
        self.beta_r = 0.88
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
    Wr[-1,0] = 1
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
    global Wr, Wb, Wo, Wi
    #Wr = generate_random_matrix(c.Nh,c.Nh,c.alpha_r,c.beta_r,distribution="one",normalization="sr",diagnal=True)
    Wr = bm_weight()
    #Wr = ring_weight()
    #Wr = small_world_weight()
    Wb = generate_random_matrix(c.Nh,c.Ny,c.alpha_b,c.beta_b,distribution="one",normalization="none")
    Wi = generate_random_matrix(c.Nh,c.Nu,c.alpha_i,c.beta_i,distribution="one",normalization="none")
    Wo = np.zeros(c.Nh * c.Ny).reshape((c.Ny, c.Nh))

def fy(h):
    return np.tanh(h)

def fyi(h):
    return np.arctanh(h)

def p2s(theta,p):
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1)

def softmax(a):
    x = np.exp(a)
    u = np.sum(x)
    return x/u

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

    idx_linear = int(c.alpha0*c.Nh)
    rs = 1
    any_hs_change = True
    count =0
    m = 0
    prev_hc = np.zeros((c.Nh))
    hx_prev = np.zeros((c.Nh))
    hs_prev = np.zeros((c.Nh))
    for n in tqdm(range(c.NN * c.MM)):
        theta = np.mod(n/c.NN,1) # (0,1)
        rs_prev = rs
        hs_prev = hs.copy()

        rs = p2s(theta,0)# 参照クロック
        us = p2s(theta,Up[m]) # エンコードされた入力
        ds = p2s(theta,Dp[m]) #
        ys = p2s(theta,yp)
        #linear = Wi@(2*us-1) + Wr@(2*p2s(theta,hp)-1) + c.alpha_s*(hs-rs)*ht

        sum = np.zeros(c.Nh)
        #sum += c.alpha_s*rs # ラッチ動作を用いないref.clockと同期させるための結合
        sum += c.alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
        #sum += c.alpha_s * hx 
        sum += Wi@(2*us-1) # 外部入力
        sum += Wr@(2*hs-1) # リカレント結合
        # sum += Wr@(2*p2s(theta,hp)-1) # リカレント結合

        #sum += Wr@prev_hc/c.NN
        #sum += Wr@(2*p2s(theta,hp)-1)
        #if mode == 0:
        #    sum += Wb@ys
        #if mode == 1:  # teacher forcing
        #    sum += Wb@ds

        hsign = 1 - 2*hs
        hx = hx + hsign*(1+np.exp(hsign*sum/c.Temp))*c.dt
        #hx[:idx_linear] = linear[:idx_linear]
        #hx[:idx_linear] += p2s(theta,lx[:idx_linear])*c.dt
        hs = np.heaviside(hx+hs-1,0)
        hx = np.fmin(np.fmax(hx,0),1)
        
        hc[(hs_prev == 1)& (hs==0)] = count
        
        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            linear = Wi@Up[m] + Wr@hp
            hp = 2*hc/c.NN-1    # デコード、カウンタの値を連続値に変換
            hp[:idx_linear] = linear[:idx_linear]

            hs[:idx_linear] = p2s(theta,linear[:idx_linear])
            #hp = (hp+linear)/2
            # hx = hc/c.NN
            # hs = np.heaviside(hx+hs-1,0)
            # hx = np.fmin(np.fmax(hx,0),1)
            #hp = 2*hc/c.NN-1    # デコード、カウンタの値を連続値に変換

            hc = np.zeros(c.Nh) #カウンタをリセット
            ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト
            yp = Wo@hp
            # record    
            Hp[m]=hp
            Yp[m]=yp
            count = 0
            m += 1
            prev_hc = hc

        #境界条件
        if n == (c.NN * c.MM-1):
            #linear = hp#Wi@ Up[m] + Wr@hp
            #linear = Wi@Up[m] + Wr@hp
            hp = 2*hc/c.NN-1    # デコード、カウンタの値を連続値に変換
            #idx_linear = int(c.alpha0*c.Nh)
            #hp[:idx_linear] = linear[:idx_linear]

            # hp = 2*hc/c.NN-1    # デコード、カウンタの値を連続値に変換

            yp = Wo@hp
            # record
            Hp[m]=hp
            Yp[m]=yp
            prev_hc = hc

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
    c.Ny = int(c.delay)
    np.random.seed(c.seed)    
    c.delay = int(c.delay) 

    ### generate data
    name_list = ["Legendre","Hermite","Chebyshev","Laguerre"]
    dist_list = ["uniform","normal","arcsine","exponential"]
    dist = dist_list[c.set]
    name = name_list[c.set]
    
    U,D = datasets(k=c.delay,n=c.degree,T = c.MM,name=name,dist=dist,seed=c.seed,new=0)
    #print(U.shape,D.shape)
    # max = np.max(np.max(abs(D)))
    # D /= max*1.01
    # U /= max*1.01
    # plt.plot(D[:,0]) 
    #plt.plot(U)
    # plt.show()
    D = D[:,:c.delay]
    #print(D.shape)

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
    #MC = np.corrcoef(Dp[c.delay:,0],Yp[c.delay:,0])[0,1]**2
    for i in range(c.delay):
        r = np.corrcoef(Dp[c.delay:,i],Yp[c.delay:,i])[0,1]
        CAPACITY.append(r**2)
    MC = sum(CAPACITY)
    # ep = 1.7*10**(-4)
    # MC = np.heaviside(MC-ep,1)*MC
    
    # SUM = np.sum((Yp-Dp)**2)
    # RMSE1 = np.sqrt(SUM/c.Ny/(c.MM-c.MM0-c.delay))

    #RMSE2 = 0
    # print("-------------"+name+","+dist+",degree = "+str(c.degree)+"-------------")
    # print(CAPACITY)
    # #print("RMSE=",RMSE1)
    # print("IPC=",MC)


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
    if c.plot: plot1()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-config", type=str)
    a = ap.parse_args()

    c=Config()
    if a.config: c=common.load_config(a)
    #degree  = c.degree
    name_list = ["Legendre","Hermite","Chebyshev","Laguerre"]
    dist_list = ["uniform","normal","arcsine","exponential"]
    
    # for c.set in range(4):
    #     c.dist = dist_list[c.set]
    #     c.name = name_list[c.set]
    #     for i in range(1,11):
    #         c.plot = 0
    #         c.degree = i
    c.name,c.dist = name_list[c.set],dist_list[c.set]

    #print(c.alpha0)
    
    
    for i in range(1,11):
        c.degree = i
        execute(c)
        plt.plot(c.CAPACITY,label="degree = "+str(c.degree))

    # plt.bar([c.alpha_i],[c.CAPACITY],bottom=prev,width=0.1,label=str(i+1))
    # prev+=c.CAPACITY
    #c.per.append([[c.alpha_i],[c.CAPACITY]]
    plt.ylabel("Capacity")
    plt.xlabel("delay")
    plt.ylim([-0.1,1.1])
    plt.xlim([-0.1,20.1])
    plt.title("cbm::"+c.name+"::"+c.dist+"::"+str(c.alpha0))
    plt.legend()
    #plt.show()
    t = common.string_now()
    common.create_directory(t+"-eps-fig")
    na = "./%s-eps-fig/%s-ipc4_cbm_fixed_in_tar_%s.eps" % (t,t,str(c.set))
    plt.savefig(na)
    plt.clf()
     
    if a.config: common.save_config(c)
    # if a.config: c=common.load_config(c)
    # execute()
    # if a.config: common.save_config(c)
