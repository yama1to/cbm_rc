from re import X
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as st
from scipy.special import factorial


def Legendre(x,degree):
    P = 1
    for n in range(1,1+degree):
        tmp =0
        for k in range(int(np.floor(n/2)+1)):
            tmp += (x**(n-2*k))*\
                ((-1)**k*factorial(2*n-2*k))/\
                    (factorial(n-k)*factorial(k)*factorial(n-2*k))
        P *= tmp / 2**n
    return P 

def Hermite(x,degree):
    P = 1
    for n in range(1,1+degree):
        tmp = factorial(n)
        for k in range(int(np.floor(n/2)+1)):
            tmp += ((2*x)**(n-2*k) * (-1)**k) / (factorial(n-2*k)*factorial(k))

    P *= tmp 
    return P 

def Chebyshev(x,degree):
    P = 1
    for n in range(1,1+degree):
        tmp = 0
        for k in range(int(np.floor(n/2)+1)):
            tmp += (factorial(n)/factorial(n-2*k)/factorial(2*k))*x**(n-2*k) * (x**2 -1)**k
        P*=tmp
    return tmp

def Laguerre(x,degree):
    P = 1
    for n in range(1,1+degree):
        tmp = factorial(n)
        for k in range(n+1):
            tmp += (-1)**k * factorial(n)*x**k/factorial(n-k)/factorial(k)**2
        P *= tmp 
    return P 

def polynomial(name,x,degree):
    if degree == 0: return np.ones(x.shape)
    if degree == 1: return x
    if name == "Legendre":
        return Legendre(x,degree)
    elif name == "Hermite":
        return Hermite(x,degree)
    elif name == "Chebyshev":
        return Chebyshev(x,degree)
    elif name == "Laguerre":
        return Laguerre(x,degree)

def datasets(k=2,n = 2,
                  T=1000,
                  name="Legendre",
                  dist="uniform",
                  #dist="exponential",
                  seed=0,
                  new=0):
    """
    k:遅延長
    n:多項式の次数
    name:使用する多項式
    dist:入力の分布
    seed:乱数のシード値
    """
    if not new:
        u = np.load("./ipc3_dir/input"+str(n)+".npy")
        d = np.load("./ipc3_dir/target"+str(n)+".npy")
        return u,d
    np.random.seed(seed)

    if dist=="normal":
        u = np.random.normal(size=(T,1))
    elif dist=="uniform":
        u = np.random.uniform(-1,1,(T,1))
    elif dist=="arcsine":
        u = st.arcsine.rvs(size=(T,1))
    elif dist=="exponential":
        u = st.expon.rvs(size=(T,1))

    
    delay = np.arange(k)  # 遅延長z 
    d = np.empty((T, len(delay)))
    
    for t in range(T):
        for k in range(len(delay)):
            y = polynomial(name,u,n)
            # if k>0:
            #     y *=prev_y
            d[t, k] = y[t-delay[k],0]  # 遅延系列
            prev_y = y


    if new:
        np.save("./ipc3_dir/input"+str(n)+".npy",arr=u)
        np.save("./ipc3_dir/target"+str(n)+".npy",arr=d)
    return u,d





if __name__=="__main__":
    for i in range(10+1):
        u,d = datasets(k=20,n = i,
                    T=1000, 
                    name="Legendre",
                    dist="uniform",
                    #dist="exponential",
                    seed=0,
                    new=1)

    # plt.plot(d[:,:])
    # plt.show()
    # print(u.shape,d.shape)