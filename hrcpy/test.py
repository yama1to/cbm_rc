from model import Input,Reservoir#,Output
from data_generater import Data
import matplotlib.pyplot as plt 
from pprint import pprint
import numpy as np
from encode_decode import encode,decode
from eval import RMSE

if  __name__ == '__main__':
#==============================================================#
#------------ INPUT TEST --------------#
    
    data = Data.sinwave(L=50,X=2,Y=2000)
    #print(data)
    input= Input(N_u = 2,
                 N_x = 200, 
                 input_scale= 1, 
                 seed=0)
    
    step = 200
    x,y = data.shape
    print(x,y)
    u_s = encode(data[0],y,step=step)

    print(u_s.shape)
    dec = decode(u_s,y,step=step)
    plt.plot(data[0])
    plt.plot(dec)
    plt.show()
    rmse = RMSE(data[0],dec)
    print(rmse)



#==============================================================#
#------------ RESERVOIR TEST ------------#
    """
    reservoir = Reservoir(
                        N_x=200,
                        density=0.1,
                        rho=0.5,
                        U=data,
                        step=200,
                        activation_func=np.tanh,
                        leaking_rate=0.1,
                        seed=0
                        )

    W = reservoir.W

    J = input(data)

    next_x = reservoir(1,J)

    pprint(next_x)
    """


#==============================================================#
#------------ OUTPUT TEST ------------#