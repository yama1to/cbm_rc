
import numpy as np
import matplotlib.pyplot as plt 


def generate_data(num=300,delay=10,logv=0.5,f=np.sin):
    num = num +delay
    s = np.random.uniform(-1,1,(num,1))
    y = np.zeros((num,1))
    
    v = np.e**logv
    for i in range(delay,num):
        y[i] = f(v * s[i-delay])

    y = y[delay:]
    s = s[delay:]

    return s,y

if __name__ == "__main__":

    u,target = generate_data()
    plt.plot(u)
    plt.plot(target)
    plt.show()
