
import math
import numpy as np
import matplotlib.pyplot as plt



def mu(y, t):
    """Implement the Ornstein–Uhlenbeck mu."""  # = \theta (\mu-Y_t)
    return c_theta * (c_mu - y)

def sigma(y, t):
    """Implement the Ornstein–Uhlenbeck sigma."""  # = \sigma
    return c_sigma

def dW(delta_t):
    """Sample a random number at each call."""
    return np.random.normal(loc=0.0, scale=np.sqrt(delta_t))



def OUdataset(N = 1000):
    num_sims = 1  # Display five runs
    N      = N-1  # Compute 1000 grid points

    t_init = 0
    t_end  = 100
    
    dt     = float(t_end - t_init) / N
    y_init = 0

    global c_sigma,c_mu,c_theta

    c_theta = 0.7
    c_mu    = 0
    c_sigma = 0.06

    ts = np.arange(t_init, t_end + dt, dt)
    ys = np.zeros(N + 1)

    ys[0] = y_init

    for _ in range(num_sims):
        for i in range(1, ts.size):
            t = t_init + (i - 1) * dt
            y = ys[i - 1]
            ys[i] = y + mu(y, t) * dt + sigma(y, t) * dW(dt)
    
    ys = ys.reshape(ys.size,1)
    return ts ,ys

def datasets(T=2000,delay_s=20):
    

    T = T+200
    _,u = OUdataset(N=T)

    delay = np.arange(delay_s)  # 遅延長
    d = np.empty((T, len(delay)))
    for k in range(len(delay)):
        for t in range(T):
            d[t, k] = u[t-delay[k]]  # 遅延系列

    T_trans = 200  # 過渡期
    train_U = u[T_trans:T].reshape(-1, 1)
    train_D = d[T_trans:T, :].reshape(-1, len(delay))
    return train_U,train_D

if __name__ == "__main__":
    #ts,ys = OUdataset()
    ts,ys = datasets(T=2200)
    plt.plot(ys)
    plt.show()
    print(ys.shape)