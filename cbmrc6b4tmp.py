# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:
import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv

import matplotlib as mpl
mpl.use('Agg')## サーバ上で画像を出力するための設定。ローカルで表示する際はコメントアウトする
import matplotlib.pyplot as plt

import sys
import copy
from arg2x import *
from generate_data_sequence import *

file_csv = "data_cbmrc6b3.csv"
file_fig1 = "data_cbmrc6b3_fig1.png"
display = 1
dataset = 2
seed=0 # 乱数生成のためのシード
id=0

NN=200
MM=300
MM0 = 50

Nu = 2   #size of input
Nh = 200 #size of dynamical reservior
Ny = 2   #size of output

Temp=1
dt=1.0/NN #0.01

#sigma_np = -5
alpha_i = 0.15 #0.2
alpha_r = 0.2 # 0.25
alpha_b = 0.
alpha_s = 0.7 # 0.6

alpha0 = 0#0.1
alpha1 = 0#-5.8

beta_i = 0.1
beta_r = 0.1
beta_b = 0.1

#tau = 2
lambda0 = 0.1

def config():
    global file_csv,file_fig1,display,dataset,seed,id,NN,Nh,alpha_i,alpha_r,alpha_b,alpha_s,alpha0,alpha1,beta_i,beta_r,beta_b,Temp,lambda0
    args = sys.argv
    for s in args:
        file_csv= arg2a(file_csv,"file_csv=",s)
        file_fig1=arg2a(file_fig1,"file_fig1=",s)
        display = arg2i(display,"display=",s)
        dataset = arg2i(dataset,"dataset=",s)
        seed    = arg2i(seed,"seed=",s)
        id      = arg2i(id,"id=",s)
        NN      = arg2i(NN, 'NN=', s)
        Nh      = arg2i(Nh, 'Nh=', s)
        alpha_i = arg2f(alpha_i,"alpha_i=",s)
        alpha_r = arg2f(alpha_r,"alpha_r=",s)
        alpha_b = arg2f(alpha_b,"alpha_b=",s)
        alpha_s = arg2f(alpha_s,"alpha_s=",s)
        alpha0  = arg2f(alpha0,"alpha0=",s)
        alpha1  = arg2f(alpha1,"alpha1=",s)
        beta_i  = arg2f(beta_i,"beta_i=",s)
        beta_r  = arg2f(beta_r,"beta_r=",s)
        beta_b  = arg2f(beta_b,"beta_b=",s)
        Temp    = arg2f(Temp,"Temp=",s)
        lambda0 = arg2f(lambda0, 'lambda0=', s)

def output():
    str="%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" \
    % (dataset,seed,id,NN,Nh,alpha_i,alpha_r,alpha_b,alpha_s,alpha0,alpha1,beta_i,beta_r,beta_b,Temp,lambda0,RMSE1,RMSE2,count_gap,overflow)
    f=open(file_csv,"a")
    f.write(str)
    f.close()

def generate_weight_matrix():
    global Wr, Wb, Wo, Wi
    ### Wr
    Wr0 = np.zeros(Nh * Nh)
    nonzeros = Nh * Nh * beta_r
    Wr0[0:int(nonzeros / 2)] = 1
    Wr0[int(nonzeros / 2):int(nonzeros)] = -1
    np.random.shuffle(Wr0)
    Wr0 = Wr0.reshape((Nh, Nh))
    v = scipy.linalg.eigvals(Wr0)
    lambda_max = max(abs(v))
    #print("WoT\n", WoT)
    #print("lambda_max",lambda_max)
    Wr = Wr0 / lambda_max * alpha_r
    E = np.identity(Nh)
    Wr = Wr + alpha0*E
    #Wr = Wr + alpha1

    Wr = Wr + alpha1/Nh
    #Wr = Wr -0.06#/Nh

    # print("lamda_max",lambda_max)
    # print("Wr:")
    # print(Wr)

    ### Wb
    Wb = np.zeros(Nh * Ny)
    Wb[0:int(Nh * Ny * beta_b / 2)] = 1
    Wb[int(Nh * Ny * beta_b / 2):int(Nh * Ny * beta_b)] = -1
    np.random.shuffle(Wb)
    Wb = Wb.reshape((Nh, Ny))
    Wb = Wb * alpha_b
    # print("Wb:")
    # print(Wb)

    ### Wi
    Wi = np.zeros(Nh * Nu)
    Wi[0:int(Nh * Nu * beta_i / 2)] = 1
    Wi[int(Nh * Nu * beta_i / 2):int(Nh * Nu * beta_i)] = -1
    np.random.shuffle(Wi)
    Wi = Wi.reshape((Nh, Nu))
    Wi = Wi * alpha_i
    # print("Wi:")
    # print("WoT\n", WoT)
    # print(Wi)Ds = np.zeros((MM*NN, Ny))
    Us = np.zeros((MM*NN, Nu))

    ### Wo
    Wo = np.zeros(Nh * Ny)
    Wo = Wo.reshape((Ny, Nh))
    Wo = Wo
    # print(Wo)

def fx(h):
    return np.tanh(h)

def fy(h):
    return np.tanh(h)

def fyi(h):
    #print("WoT\n", WoT)
    return np.arctanh(h)
    #return -np.log(1.0/h-1.0)
def fr(h):
    return np.fmax(0, h)

def fsgm(h):
    return 1.0/(1.0+np.exp(-h))

def flgt(h):
    return np.log(1/(1-h))

def update_s(x,s,N):
    for i in range(N):
        if x[i]>=1:
            x[i]=1
            s[i]=1
        if x[i]<=0:
            x[i]=0
            s[i]=0
def p2s(theta,p):
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1)
def Phi(x):
    return np.heaviside( np.sin(2*np.pi*x),1)

def run_network(mode):
    global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y, Us, Ds,Rs
    Hp = np.zeros((MM, Nh))
    Hx = np.zeros((MM*NN, Nh))
    Hs = np.zeros((MM*NN, Nh))
    hsign = np.zeros(Nh)

    #hx = np.zeros(Nh)
    hx = np.random.uniform(0.5,1.0,Nh) # [0,1]の連続値
    #hx = np.ones(Nh)*0.5
    hs = np.random.randint(0,1,Nh)
    #hs = np.zeros(Nh) # {0,1}の２値

    hs_prev = np.zeros(Nh)
    hc = np.zeros(Nh) # ref.clockに対する位相差を求めるためのカウント
    hf = np.zeros(Nh)
    hp = np.zeros(Nh) # [-1,1]の連続値
    ht = np.zeros(Nh) # {0,1}

    Yp = np.zeros((MM, Ny))
    Yx = np.zeros((MM*NN, Ny))
    Ys = np.zeros((MM*NN, Ny))
    #ysign = np.zeros(Ny)
    yp = np.zeros(Ny)
    yx = np.zeros(Ny)
    ys = np.zeros(Ny)
    #yc = np.zeros(Ny)

    Us = np.zeros((MM*NN, Nu))
    Ds = np.zeros((MM*NN, Ny))
    Rs = np.zeros((MM*NN, 1))

    rs = 1
    rs_prev = 0
    count=0
    global count_of
    global overflow
    count_of=0
    count_num=0
    m=0
    for n in range(NN*MM):
        theta = np.mod(n/NN,1) # (0,1)
        rs_prev = rs
        #rs = p2s(theta,0)
        #us = p2s(theta,Up[m])
        #ds = p2s(theta,Dp[m])
        #ys = p2s(theta,yp)
        rs = Phi(theta)
        us = Phi(theta - 0.5*Up[m])
        ds = Phi(theta - 0.5*Dp[m])
        ys = Phi(theta - 0.5*yp)

        sum = np.zeros(Nh)
        sum += alpha_s*(hs-rs)*ht # ref.clockと同期させるための結合
        sum += Wi@(2*us-1) # 外部入力
        sum += Wr@(2*hs-1) # リカレント結合

        #if mode == 0:
        #    sum += Wb@ys
        #if mode == 1:  # teacher forcing
        #    sum += Wb@ds

        hsign = 1 - 2*hs
        hx = hx + hsign*(1.0+np.exp(hsign*sum/Temp))*dt
        hs_prev = hs.copy()
        update_s(hx,hs,Nh)

        # hs の立ち下がりで count の値を hc に保持する。
        for i in range(Nh):
            if hs_prev[i]==1 and hs[i]==0:
                hc[i]=count
                hf[i]+=1 # hsの立ち下がりの回数をカウント：普通は１回
        #print(n,n%NN,l,hs_prev[0],hs[0],hc[0])
        #if m<3 or (m>198 and m<204):print("n=%3d m=%3d rs=%d rs_prev=%d %3d theta=%f MM=%d"%(n,m,rs,rs_prev,count,theta,MM))

        count = count + 1

        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            hp = 2*hc/NN-1
            ht = 2*hs-1
            yp = fy(Wo@hp)
            #yp=fsgm(Wo@hp)
            count=0
            if m>100:
                count_of += np.sum( np.abs(hf-1) ) # hsの立ち下がりが１回でなかった回数
                count_num += 1
            #print("m:",m,"count_of",count_of)
            hf = np.zeros(Nh) # hsの立ち下がりのカウントをリセット
            # record
            Hp[m]=hp
            Yp[m]=yp
            m+=1

        # record
        Rs[n]=rs
        Hx[n]=hx
        Hs[n]=hs
        Yx[n]=yx
        Ys[n]=ys
        Us[n]=us
        Ds[n]=ds

    # 不連続な値の変化を検出する。
    overflow = count_of/count_num #/Nh
    #print("overflow:",overflow)

    global count_gap
    count_gap = 0
    for m in range(2,MM-1):
        tmp = np.sum( np.heaviside( np.fabs(Hp[m+1]-Hp[m]) - 0.6 ,0))
        count_gap += tmp
        #print(tmp)

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = Hp[MM0:, :]
    invD = fyi(Dp)
    G = invD[MM0:, :]

    #print("Hp\n",Hp)
    #print("M\n",M)

    ### Ridge regression
    E = np.identity(Nh)
    TMP1 = inv(M.T@M + lambda0 * E)
    WoT = TMP1@M.T@G
    Wo = WoT.T
    #print("WoT\n", WoT)

def test_network():
    run_network(0)

def plot(data):
    fig, ax = plt.subplots(1,1)
    ax.cla()
    ax.plot(data)
    plt.show()

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
    ax.set_ylim(-1,1)
    ax.plot(Yp)

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.set_title("Dp")
    ax.set_ylim(-1,1)
    ax.plot(Dp)

    plt.show()
    plt.savefig(file_fig1)

def plot2():
    fig=plt.figure(figsize=(20, 12))
    Nr=3
    t1=100
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("U")
    ax.plot(Up[t1:])

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("Hp")
    ax.plot(Hp[t1:,:100])

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Y, Ytarget")
    ax.set_ylim(-1,1)
    ax.plot(Yp[t1:])
    ax.plot(Dp[t1:],'--')

    plt.show()
    plt.savefig(file_fig1)

def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2
    if seed>=0:
        np.random.seed(seed)
    generate_weight_matrix()

    ### generate data
    if dataset==1:
        MM1=300 # length of training data
        MM2=400 # length of test data
        D, U = generate_simple_sinusoidal(MM1+MM2)
    if dataset==2:
        MM1=300 # length of training data
        MM2=300 # length of test data
        D, U = generate_complex_sinusoidal(MM1+MM2)
    if dataset==3:
        MM1=1000 # length of training data
        MM2=1000 # length of test data
        D, U = generate_coupled_lorentz(MM1+MM2)
    D1 = D[0:MM1]
    U1 = U[0:MM1]
    D2 = D[MM1:MM1+MM2]
    U2 = U[MM1:MM1+MM2]

    ### training
    #print("training...")
    MM=MM1
    Dp = np.tanh(D1)
    Up = np.tanh(U1)
    train_network()

    ### test
    #print("test...")
    MM=MM2
    Dp = np.tanh(D2)
    Up = np.tanh(U2)
    test_network()

    ### evaluation
    sum=0
    for j in range(MM0,MM):
        sum += (Yp[j] - Dp[j])**2
    SUM=np.sum(sum)
    RMSE1 = np.sqrt(SUM/Ny/(MM-MM0))
    RMSE2 = 0

    print("RMSE1:",RMSE1,"count_gap:",count_gap,"overflow:",overflow)

    if display :
        plot2()

if __name__ == "__main__":
    config()
    execute()
    output()
