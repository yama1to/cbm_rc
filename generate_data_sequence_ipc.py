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


def datasets(n_k=np.array([[1,1],
                         [1,2]]),
                  T=1000,
                  name="Legendre",
                  dist="normal",
                  #dist="exponential",
                  seed=0):
    """
    dist =     ["normal",
                "uniform",
                "arcsine",
                "exponential"]
    """
    max = np.max(n_k[:,1])
    T += max
    V,_ = n_k.shape
    np.random.seed(seed)

    if dist=="normal":
        u = np.random.normal(size=(T,1))
    if dist=="uniform":
        u = np.random.uniform(-1,1,(T,1))
    if dist=="arcsine":
        u = st.arcsine.rvs(size=(T,1))
    if dist=="exponential":
        u = st.expon.rvs(size=(T,1))
    # plt.hist(u,bins = np.arange(-1,1,0.01))
    # plt.show()
    #u = u.reshape(T,1)
    d = np.zeros((T,1))

    
    for l in range(T-max):
        y = 1
        for i in range(V):
            [n,k] = n_k[i]
            y *= polynomial(name=name,x=u[l+k,0],degree=n)

        d[l,0] = y

    u = u[:T-max]
    d = d[:T-max]
    d = d.reshape(-1,1)
    #print(u.shape,d.shape)

    return u,d
    plt.plot(u)
    plt.plot(d)
    plt.show()



if __name__=="__main__":
    # func = polynomial(n=2,name="Hermite")
    # a = func(3)
    # print(a)
    # data = np.random.uniform(0,1,(100000,1))
    # d = func(data)

    # #plt.plot(d)
    # plt.hist(d,bins=np.arange(-10,10,0.1))
    # plt.show()

    u,d= datasets()
    #print(u,d)
    plt.plot(d)
    plt.show()