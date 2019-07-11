#!/usr/bin/python
# Copyright (c) 2014-2017 Katori Lab. All Rights Reserved
# NOTE: matplot minimal example

import numpy as np
import matplotlib.pyplot as plt

def generate_simple_sinusoidal(MM, Nu=2, Ny=2):
    D = np.zeros((MM, Ny))
    U = np.zeros((MM, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    for n in range(MM):
        t = 0.25 * n  # 0.5*n
        d = np.sin(t + cy) * 0.8
        #d = np.sin(t+cy)*np.exp(-0.1*(t-10)**2)*0.5
        u = np.sin(t*0.5 + cu) * 0.8
        D[n, :] = d
        U[n, :] = u
    return (D, U)

def generate_complex_sinusoidal(MM, Nu=2, Ny=2):
    D = np.zeros((MM, Ny))
    U = np.zeros((MM, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    for n in range(MM):
        t = 0.25 * n  # 0.5*n
        #d = np.sin(t + cy) * 0.8
        d = np.sin(0.5*t + cy) * 0.8 + np.sin(1.5*t + cy) * 0.5
        u = np.sin(t*0.5 + cu) * 0.8
        D[n, :] = d
        U[n, :] = u
    return (D, U)
def generate_coupled_lorentz(MM, Nu=2, Ny=2, K=8,tau1=1.0,tau2=1.0):
    """
    driver and response system of Lorentz system
    MM:length of data
    K:coupling strength
    """
    assert Nu == 2 and Ny == 2, "(Nu,Ny) should be (2,2)"
    x1 = 10
    y1 = 0
    z1 = 0
    x2 = 9
    y2 = 0
    z2 = 0
    dt = 0.01
    sigma = 10
    r = 28
    b = 8.0/3.0

    sampling_interval = 5
    scale = 0.05

    D = np.zeros((MM, Ny))
    U = np.zeros((MM, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    i = 0
    n = 0
    while n < MM:
        # drive system
        _x1 = x1+(-sigma*x1 + sigma*y1)*dt/tau1
        _y1 = y1+(-x1*z1 + r*x1-y1)*dt/tau1
        _z1 = z1+(x1*y1 - b*z1)*dt/tau1
        # response system
        _x2 = x2+(-sigma*x2 + sigma*y2 + K*(x1-x2))*dt/tau2
        _y2 = y2+(-x2*z2 + r*x2-y2)*dt/tau2
        _z2 = z2+(x2*y2 - b*z2)*dt/tau2
        # update
        x1 = _x1
        y1 = _y1
        z1 = _z1
        x2 = _x2
        y2 = _y2
        z2 = _z2
        if i % sampling_interval == 0:
            U[n, :] = x1*scale, y1*scale
            D[n, :] = x2*scale, y2*scale
            n += 1
        i += 1
    return (D, U)

def generate_coupled_lorentz2(MM, Nu=1, Ny=2, K=8,tau1=1.0,tau2=1.0):
    """
    driver and response system of Lorentz system
    MM:length of data
    K:coupling strength
    """
    #assert Nu == 2 and Ny == 2, "(Nu,Ny) should be (2,2)"
    x1 = 10
    y1 = 0
    z1 = 0
    x2 = 9
    y2 = 0
    z2 = 0
    dt = 0.01
    sigma = 10
    r = 28
    b = 8.0/3.0

    sampling_interval = 5
    scale = 0.05

    D = np.zeros((MM, Ny))
    U = np.zeros((MM, Nu))
    cy = np.linspace(0, 1, Ny)
    cu = np.linspace(0, 1, Nu)
    i = 0
    n = 0
    while n < MM:
        # drive system
        x1 = 7 + 5*np.cos(2*np.pi*n/4000)
        # response system
        sigma = x1
        _x2 = x2+(-sigma*x2 + sigma*y2 )*dt/tau2
        _y2 = y2+(-x2*z2 + r*x2 - y2 )*dt/tau2
        _z2 = z2+(x2*y2 - b*z2 )*dt/tau2
        # update
        x2 = _x2
        y2 = _y2
        z2 = _z2
        if i % sampling_interval == 0:
            U[n, :] = x1*scale
            D[n, :] = x2*scale, z2*scale
            n += 1
        i += 1
    return (D, U)

def generate_random_spike(MM, Nu=2):
    D = np.zeros((MM, Nu))
    U = np.zeros((MM, Nu))
    u = np.zeros(Nu)
    d = np.zeros(Nu)

    for n in range(MM):
        for i in range(Nu):
            p = np.random.uniform(0,1)
            if p<0.5:
                u[i]=1
            else:
                u[i]=0

        #print(u)
        D[n, :] = u
        U[n, :] = u
    return (D, U)

if __name__ == "__main__":
    #D, U = generate_simple_sinusoidal(500)
    #D, U = generate_complex_sinusoidal(500)
    #D, U = generate_coupled_lorentz2(2000,K=1,tau1=4)
    D, U = generate_random_spike(200)

    plt.subplot(2, 1, 1)
    plt.plot(U)
    plt.ylabel('U')

    plt.subplot(2, 1, 2)
    plt.plot(D)
    plt.ylabel('D')
    plt.show()
    # plt.savefig("test.png")
