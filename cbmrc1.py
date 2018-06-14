import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt

NN=100
MM=100

Nu = 2   #size of input
Nh = 100 #size of dynamical reservior
Ny = 2   #size of output

sigma_np = -5
alpha_r = 0.8
alpha_b = 0.8
alpha_i = 0.8
beta_r = 0.1
beta_b = 0.1
beta_i = 0.1
alpha0 = 0.7
tau = 2
lambda0 = 0.1

def generate_data_sequence():
    D = np.zeros((MM, Ny))
    U = np.zeros((MM, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    for n in range(MM):
        t = 0.1 * n
        d = np.sin(t + cy) * 0.8
        # d=np.sin(t+c)*np.exp(-0.1*(t-10)**2)*0.5
        u = np.sin(t*0.3 + cu) * 0.8
        D[n, :] = d
        U[n, :] = u
    return (D, U)

def generate_s_sequence():
    Ds = np.zeros((MM, Ny, NN))
    Us = np.zeros((MM, Nu, NN))
    for m in range(MM):
        for p in range(2):
            Ds[m][p][0:int(NN*Dp[m][p])] = 1
            Ds[m][p][int(NN*Dp[m][p]):int(NN)] = 0
            Us[m][p][0:int(NN*Up[m][p])] = 1
            Us[m][p][int(NN*Up[m][p]):int(NN)] = 0
    return (Ds, Us)

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
    Wr = Wr0 / lambda_max * alpha_r

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
    # print(Wi)

    ### Wo
    Wo = np.ones(MM * Ny)
    Wo = Wo.reshape((Ny, MM))
    Wo = Wo
    # print(Wo)

def fx(h):
    return np.tanh(h)


def fy(h):
    return np.tanh(h)


def fyi(h):
    return np.arctanh(h)


def fr(h):
    return np.fmax(0, h)

def fsgm(h):
    return 1.0/(1.0+np.exp(-h))

def flog(h):
    return np.log(1/h-1)

def update(hs, hx):
    for i in range(Nh):
        for j in range(NN):
            if(hx[i][j] == 0):
                hs[i][j] == 0
            if(hx[i][j] == 1):
                hs[i][j] == 1

def run_network(mode):
    global Hx, Hs, Hp, Y, Ys, Yp
    Hx = np.zeros((MM, Nh, NN))
    Hs = np.zeros((MM, Nh, NN))
    Hp = np.zeros((MM, Nh))
    Ys = np.zeros((MM, Ny, NN))
    Yp = np.zeros((MM, Ny))
    Y = np.zeros((MM, Ny))

    m = 0
    hx = np.random.uniform(0, 1, (Nh, NN))
    hs = np.random.randint(0, 2, (Nh, NN))
    hp = np.random.uniform(0, 1, Nh)
    ys = np.random.randint(0, 2, (Ny, NN))
    yp = np.random.uniform(0, 1, Ny)
    y = np.random.uniform(-1, 2, Ny)
    a = np.ones((Nh, NN))
    Hx[m, :] = hx
    Hs[m, :] = hs
    Hp[m, :] = np.sum(hs)/NN
    Ys[m, :] = ys
    Yp[m, :] = np.sum(ys)/NN
    Y[m, :] = y
    for m in range(MM - 1):
        sum = np.zeros((Nh, NN))
        us = Us[m, :]
        sum += Wi@us
        sum += Wr@hs
        if mode == 0:
            sum += Wb@ys
        if mode == 1:  # teacher forcing
            ds = Ds[m, :]
            sum += Wb@ds
        sign = a - 2*hs
        print(sum)
        hx = hx + sign*(1+np.exp(sign*sum))
        hx = np.where(hx<0, 0, hx)
        hx = np.where(hx>1, 1, hx)
        update(hs, hx)
        for h in range(Nh):
            hp[h] = np.sum(hs[h])/NN
        yp = Wo@hp
        ys[0][0:int(NN*yp[0])] = 1
        ys[0][int(NN*yp[0]):int(NN)] = 0
        ys[1][0:int(NN*yp[1])] = 1
        ys[1][int(NN*yp[1]):int(NN)] = 0

        Hs[m + 1, :] = hs
        Hp[m + 1, :] = hp
        Ys[m + 1, :] = ys

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = Hp[MM:, :]
    invD = fyi(Dp)
    G = invD[MM:, :]

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

def plot2():
    fig=plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax1.cla()
    ax1.plot(U)
    ax2 = fig.add_subplot(3,1,2)
    ax2.cla()
    ax2.plot(D)
    ax3 = fig.add_subplot(3,1,3)
    ax3.cla()
    ax3.plot(Yp)
    plt.show()

def execute():
    global D,Ds,Dp,U,Us,Up
    generate_weight_matrix()
    D, U = generate_data_sequence()
    Dp = fsgm(D)
    Up = fsgm(U)
    Ds, Us = generate_s_sequence()

    train_network()
    test_network()
    plot2()

if __name__ == "__main__":
    execute()
