# Copyright (c) 2018 Katori lab. All Rights Reserved
# NOTE:
import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt
import sys
import copy
from arg2x import *
from generate_data_sequence import *




NN=200
MM=50
MM0 = 0 # T0 of T1 - T0 -1

Nu = 1  #size of input
Nh = 40 #size of dynamical reservior
Ny = 1   #size of output

Temp=1
dt=1.0/NN #0.01

alpha_i = 0.15
alpha_r = 0.1
alpha_b = 0.
alpha_s = 0.85

alpha0 = 0#0.1
alpha1 = 0#-5.8

beta_i = 0.1
beta_r = 0.1
beta_b = 0.1
#rho = 0.1

#tau = 2
lambda0 = 0.1

id = 0
ex = 'ex'
file = "XORtask.csv"
seed=0
display=1
#np.random.seed(seed=seed)print("MIN***MM:",minMM,"BER:",minber)
def config():
    global ex,file,display,seed,id,NN,Nh,alpha_i,alpha_r,alpha_b,alpha_s,alpha0,alpha1,beta_i,beta_r,beta_b,Temp,lambda0,rho
    args = sys.argv
    for s in args:
        ex      = arg2a(ex, 'ex=', s)
        file    = arg2a(file,"file=",s)
        display = arg2i(display,"display=",s)
        rho     = arg2i(rho,"rho=",s)
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
    str="%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n" \
    % (seed,id,NN,Nh,alpha_i,alpha_r,alpha_b,alpha_s,alpha0,alpha1,beta_i,beta_r,beta_b,Temp,lambda0,RMSE1,RMSE2,BER)
    f=open(file,"a")
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
    Wr = Wr0 / lambda_max * alpha_r
    E = np.identity(Nh)
    Wr = Wr + alpha0*E
    #Wr = Wr + alpha1

    Wr = Wr + alpha1/Nh
    
    #print("--------------------------------")
    eigv_list = np.linalg.eig(Wr)[0]
    sp_radius = np.max(np.abs(eigv_list))
    print(sp_radius)
    #"""
    #rho = 0.9
    # 指定のスペクトル半径rhoに合わせてスケーリング
   #Wr *= rho / sp_radius

    #eigv_list = np.linalg.eig(Wr)[0]
    #sp_radius = np.max(np.abs(eigv_list))
    #print(sp_radius)

    #print("--------------------------------")
    #"""
    #Wr = Wr -0.06#/Nh

    # print("lamda_max",lambda_max)
    # print("Wr:")
    # print(Wr)

    ### Wb
    Wb = np.zeros(Nh * Ny)
    Wb[0:int(Nh * Ny * beta_b / 2)] = 1
    Wb[int(Nh * Ny * beta_b / 2):int(Nh * Ny * beta_b)] = -1 
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

def run_network(mode):
    global Hx, Hs, Hp, Y, Yx, Ys, Yp, Y, Us, Ds,Rs
    Hp = np.zeros((MM, Nh))
    Hx = np.zeros((MM*NN, Nh))
    Hs = np.zeros((MM*NN, Nh))
    hsign = np.zeros(Nh)
    #hx = np.zeros(Nh)
    hx = np.random.uniform(0,1,Nh) # [0,1]の連続値
    hs = np.zeros(Nh) # {0,1}の２値
    hs_prev = np.zeros(Nh)
    hc = np.zeros(Nh) # ref.clockに対する位相差を求めるためのカウント
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
    m=0
    #print(NN)
    for n in range(NN*MM):
        theta = np.mod(n/NN,1) # (0,1)
        rs_prev = rs
        rs = p2s(theta,0)
        us = p2s(theta,Up[m])
        ds = p2s(theta,Dp[m])
        ys = p2s(theta,yp)

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
    global count_gap
    count_gap = 0
    for m in range(2,MM-1):
        tmp = np.sum( np.heaviside( np.fabs(Hp[m+1]-Hp[m]) - 0.6 ,0))
        count_gap += tmp
        #print(tmp)

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing
    #print(Hp)
    M = Hp[MM0:, :]
    #print(M)
    invD = fyi(Dp)
    G = invD[MM0:, :]

    #print("Hp\n",Hp)
    #print("M\n",M)

    ### Ridge regression
    E = np.identity(Nh)
    TMP1 = inv(M.T@M + lambda0 * E)
    #print(TMP1.shape, M.T.shape,G.shape)
    WoT = TMP1@M.T@G
    Wo = WoT.T
    
    #print(Wo)
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
    Nr=6+1
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
    ax.set_title("train_y,Ybin")
    ax.plot(train_Y[0:T-tau-k+1])
    ax.plot(Ybin[0:T-tau-k+1])

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.set_title("Dp")
    ax.plot(Dp)

    ax = fig.add_subplot(Nr,1,7)
    ax.cla()
    ax.set_title("Ybin,d")
    ax.plot(d[tau+k-1:T,0],label ="d")
    #ax.plot(D,label ="D")
    ax.plot(Ybin[0:T-tau-k+1],label ="Ybinary")
    ax.legend()
   # plt.plot(train_Y)
    #plt.plot(Ybin[0:T-tau-k+1])
    #"plt.plot()
    #plt.show()

    plt.show()


def execute():
    global D,Ds,Dp,U,Us,Up,Rs,R2s,MM
    global RMSE1,RMSE2
    global Ybin ,BER,train_Y,d
    global tau,k,T
    generate_weight_matrix()

    
    
    #D,U,d,u = generate_XOR(MM+2)
    #print(D.shape,U.shape,d.shape)
    T=MM+6
    
    tau = 4
    k=3
    D,U,d,u = generate_PARITY(T,tau,k)
    print(D.shape,U.shape,d.shape)

    ### generate data
    MM1=MM # length of training data
    MM2=MM # length of test data
    MM=MM1
    Dp = np.tanh(D)
    Up = np.tanh(U)
    train_network()
    
    MM=MM2
    Dp = np.tanh(D)
    Up = np.tanh(U)
    test_network()
    #print(D.T)

    RMSE1,RMSE2=0,0
    """
    ### evaluation
    sum=0
    for j in range(MM0,MM):
        sum += (Yp[j] - Dp[j])**2
    SUM=np.sum(sum)
    RMSE1 = np.sqrt(SUM/Ny/(MM-MM0))
    RMSE2 = 0
    print(RMSE1,count_gap)
"""
    T =MM+2
    # 評価（ビット誤り率, BER）
    train_Y_binary = np.zeros(T-tau-k+1)
    #print(Yp)
    train_Y = Yp

    rang = 1
    #plt.plot(train_Y)
    train_Y_binary = np.zeros(T-tau-k+1)
    for n in range(T-tau-k+1):
        if train_Y[n, 0] <= rang/2:
            train_Y_binary[n] = 0
        else:
            train_Y_binary[n] = rang
        

    Ybin = train_Y_binary
    #print(train_Y_binary.shape,d.shape)
    #print(T-tau-k+1,tau+k-1)
    BER = np.linalg.norm(Ybin[0:T-tau-k+1]-d[tau+k-1:T,0],1)/(T-tau-k+1)

    #BER = np.linalg.norm(Ybin[0:T-tau]-d[tau:T,0], 1)/(T-tau)
    print('BER =', BER)
    

    ### evaluation
    """
    sum=0
    for j in range(T-tau):
        sum += (Ybin[j] - d[j+2])**2
    SUM=np.sum(sum)
    RMSE1 = np.sqrt(SUM/Ny/(MM-MM0))
    RMSE2 = 0
    print("RMSE = ",RMSE1)
    """
   
    #ber_list.append(BER)
    
    #if display :
    #   plot1()



def param_sp_alpha_():
    global alpha_i,alpha_r,alpha_s,Temp,BER,minidx
    

    bermin = 999999
    minidx = [1,1,1,1,1]
    x=0
    for i in range(20):
        alpha_i = i/20#+0.0000001
        for j in range(1,20):
            alpha_r = j/20#+0.0000001
            for k in range(20):
                alpha_s = k/20#+0.0000001
                for l in range(1):
                    Temp = 1
                    execute()
                    print(x)
                    x+= 1
                    print(minidx)
                    if bermin > BER:
                        bermin = BER
                        minidx = [alpha_i,alpha_r,alpha_s,Temp,BER]
                        print("更新: ",minidx)
    print("END")
    print(minidx)



if __name__ == "__main__":
    """ber_list = []
    config()
    execute()
    output()
    """
    
    
    """
    ber_list = []
    param_NN()
    """
    """
    ber_list = []
    param_sp()
    """
    """
    ber_list = []
    param_sp_NN()
    """
    
    param_sp_alpha_()
