#encode_decode.py
import numpy as np
import matplotlib.pyplot as plt 
from data_generater import *

def clock(t,step):
    t_s = np.zeros((t*step))
    for i in range(t*step):
        dt = i/step
        temp = np.sin(2*np.pi*(dt))
        t_s[i] = np.heaviside(temp,1)
    return t_s

def encode(u,step=200):
    x,y = u.shape
    u_s = np.zeros((x,y*step))
    for X in range(x):
        for i in range(y*step):
            dt = i/step
            temp = np.sin(2*np.pi*(dt-u[X,np.floor(dt).astype(int)]/2))
            #temp = np.sin(2*np.pi*(dt-u[np.floor(dt).astype(int)]))
            u_s[X,i] = np.heaviside(temp,1)
    return u_s


def decode(u_s,step):
    x,y = u_s.shape
    t = y//step
    print(x,t)
    dec = np.zeros((x,t))
    for X in range(x):
        for i in range(t):
            R = 0
            for j in range(step):
                dt = j/step
                if R:
                    if u_s[X,i*step+j] == 0:
                        fallingTime = dt
                        break
                #print(i,j,step,i*step+j)
                if u_s[X,i*step+j] == 1:
                    R = 1
            dec[X,i] = 2*(fallingTime) - 1
    return dec 


if __name__ == '__main__':
    step = 200
    x,y = 1,200
    t = y
    u = Data.sinwave(50,x,y)

    cl = clock(t,step)
    us = encode(u,step)
    dec = decode(us,step)

    plt.plot(cl)
    plt.plot(us)
    plt.show()

    plt.plot(u[0],label="u")
    plt.plot(list(range(y)),dec[0],label="dec")
    plt.legend()
    plt.show()
        




