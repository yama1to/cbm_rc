import numpy as np
import scipy.linalg
from numpy.linalg import svd, inv, pinv
import matplotlib.pyplot as plt

T1 = 200
T0 = 5

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
    D = np.zeros((T1, Ny))
    U = np.zeros((T1, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    for n in range(T1):
        t = 0.1 * n
        d = np.sin(t + cy) * 0.8
        # d=np.sin(t+c)*np.exp(-0.1*(t-10)**2)*0.5
        u = np.sin(t*0.3 + cu) * 0.8
        D[n, :] = d
        U[n, :] = u
    return (D, U)

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
    Wo = np.ones(Ny * Nh)
    Wo = Wo.reshape((Ny, Nh))
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

def flgt(h):
    return np.log(1/1-h)

def run_network(mode):
    global Hp, Yp
    Hp = np.zeros((T1, Nh))
    Yp = np.zeros((T1, Ny))

    m = 0
    hp = np.random.uniform(0, 1, Nh)
    yp = np.random.uniform(0, 1, Ny)

    Hp[m, :] = hp
    Yp[m, :] = yp
    for m in range(T1 - 1):
        sum = np.zeros(Nh)
        up = Up[m, :]
        sum += Wi@up
        sum += Wr@hp
        if mode == 0:
            sum += Wb@yp
        if mode == 1:  # teacher forcing
            dp = Dp[m, :]
            sum += Wb@dp
        hp = hp + 1.0 / tau * (-alpha0 * hp + fx(sum))
        yp = fy(Wo@hp)

        Hp[m + 1, :] = hp
        Yp[m + 1, :] = yp

def train_network():
    global Wo

    run_network(1) # run netwrok with teacher forcing

    M = Hp[T1:, :]
    invD = fyi(Dp)
    G = invD[T1:, :]

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
    ax1 = fig.add_subplot(4,1,1)
    ax1.cla()
    ax1.plot(Dp)
    ax2 = fig.add_subplot(4,1,2)
    ax2.cla()
    ax2.plot(Up)
    ax3 = fig.add_subplot(4,1,3)
    ax3.cla()
    ax3.plot(Hp)
    ax4 = fig.add_subplot(4,1,4)
    ax4.cla()
    ax4.plot(Yp)
    plt.show()

def execute():
    global D,Dp,U,Up
    generate_weight_matrix()
    D, U = generate_data_sequence()
    Dp = fsgm(D)
    Up = fsgm(U)
    print(Dp)
    print(Up)


    train_network()
    test_network()
    plot2()

if __name__ == "__main__":
    execute()
