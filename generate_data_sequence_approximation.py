
import numpy as np
import matplotlib.pyplot as plt 


def generate_data(num=500,delay=0,logv=1,f=np.sin):

    if f==1 or f=="sin"     :f=np.sin
    if f==2 or f=="tan"     :f=np.tan 
    if f==3 or f=="x(1-x^2)":f=lambda x: x*(1-x**2)

    delay = int(delay)
    num = num + delay
    s = np.random.uniform(-1,1,(num,1))
    y = np.zeros((num,1))
    
    v = np.e**logv
    for i in range(delay,num):
        y[i] = f(v * s[i-delay])

    y = y[delay:]
    s = s[delay:]

    return s,y

if __name__ == "__main__":

    u,target = generate_data(delay=2,logv=3,f=3)
    print(u.shape,target.shape)
    plt.plot(u)
    plt.plot(target)
    plt.show()
