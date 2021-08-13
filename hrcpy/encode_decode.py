#encode_decode.py

import numpy as np
import matplotlib.pyplot as plt 


step = 100
t = 100
u = list(np.sin(i) for i in np.linspace(0, 2*np.pi, t))


def clock(t,step):
    t_s = np.zeros((t*step))
    for i in range(t*step):
        dt = i/step
        temp = np.sin(2*np.pi*(dt))
        t_s[i] = np.heaviside(temp,1)
    return t_s

def u_s(t,u,step):
    u_s = np.zeros((t*step))
    for i in range(t*step):
        dt = i/step
        temp = np.sin(2*np.pi*(dt-u[np.floor(dt).astype(int)]/2))
        u_s[i] = np.heaviside(temp,1)
    return u_s




def decode(u_s,t,step):
    dec = np.zeros((t))
    
    for i in range(t):
        R = 0
        for j in range(step):
            dt = 0.01
            dt = j*dt
            time = i + dt
            if R :
                if u_s[i*step+j] == 0:
                    fallingTime = time
                    break
            if u_s[i*step+j] == 1:
                R = 1
        dec[i] = 2*(fallingTime - i) - 1

        """
        i=27で変になる。
        そのためのデバッグ用
        """
        if dec[i] < -10:
            print("---------------------")
            print(i,j,time,fallingTime , i)
            print("---------------------")
        else:
            print(i,j,time,fallingTime , i)
    return dec 

cl = clock(t,step)
us = u_s(t,u,step)
dec = decode(us,t,step)

plt.plot(cl)
plt.plot(us)
plt.show()



plt.plot(u)
plt.plot(dec)
plt.show()
    




