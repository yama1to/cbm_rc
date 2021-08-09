from model import Input,Reservoir#,Output
from data_generater import Data
import matplotlib.pyplot as plt 
from pprint import pprint
import numpy as np


if  __name__ == '__main__':
#==============================================================#
#------------ INPUT TEST --------------#
    
    data = Data.sinwave(L=50,X=2,Y=2000)
    #print(data)
    Input= Input(N_u = 2,
                N_x=200, 
                input_scale= 1, 
                seed=0)
    u_s = Input.__call__(data)
    print(u_s.shape)

    x = list(range(10000))
    plt.plot(x,u_s[0,:10000])
    plt.show()
#==============================================================#
#------------ RESERVOIR TEST ------------#
    reservoir = Reservoir(
                        N_x=200,
                        density=0.1,
                        rho=0.5,
                        activation_func=np.tanh,
                        leaking_rate=0.1,
                        seed=0
                        )

    W = reservoir.make_connection()
    pprint(W)
    

#==============================================================#
#------------ OUTPUT TEST ------------#