import matplotlib.pyplot as plt 
from pprint import pprint
import numpy as np
from input import Input
from reservoir import Reservoir
from output import Output
from data_generater import Data
from encode_decode import *
from eval import *
from plot1 import plot1

if __name__=='__main__':
    N_u = 1
    N_x = 100
    N_y = 1
    learningTime = 300
    step = 200
    rho = 0.25
    activation_func = np.tanh
    seed = 0

    T = 1
    T0 = 50

    lam = 0.1
    
    alpha0  = 0.0
    alpha1  = 0.0
    alpha_i = 0.2
    alpha_r = 0.25
    alpha_s = 0.6

    beta_i  = 0.1
    beta_r  = 0.1

    # generate data
    data = Data.sinwave(L=50,X=N_u,Y=learningTime)
    #data[1] = Data.sinwave(L=30,X=N_u-1,Y=leaningTime)
    target = Data.sinwave(L=50,X=N_y,Y=learningTime-T0)

    input = Input(N_u = N_u,
                 N_x = N_x, 
                 u = data,
                 step = step,
                 alpha_i = alpha_i,
                 beta_i = beta_i,
                 seed=seed)
    
    reservoir = Reservoir(N_x,
                        data,
                        step,
                        rho,
                        activation_func,
                        alpha0=alpha0,
                        alpha1 = alpha1,
                        alpha_r=alpha_r,
                        alpha_s = alpha_s,
                        beta_r=beta_r,
                        T = 1,
                        seed=seed)

    #print(max(abs(np.linalg.eigvals(reservoir.W))))
    time = step * learningTime

    # train
    for i in range(0,time-1):
        reservoir(i,input)
    
    #decode
    r = decode(reservoir.s.T,step)
    

    output = Output(r=r,
                    N_x=N_x,
                    target = target,
                    labmda=lam,
                    T0=T0)
    
    # output
    y = output()

    # eval
    rmse = RMSE(y[0,T0:],target[0])
    print("rmse:",str(rmse))


    u_s = input.u_s
    #plot
    plot1(u=data.T,u_s=u_s.T,r_x=reservoir.x.T,
            r_decoded=r.T, output=y[:,T0:].T,target=target.T)
    



   