from re import X
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as st


def polynomial(n=2,name="Legendre"):
    """
    nとnameに応じた関数を返します.
    input: n , name 

    n : 0,1,2
    name: Legendre,Hermite,Chebyshev,Laguerre

    return function
    """
    assert 0<=n<=2,"Just n gets 0,1 or 2."
    if n==0:
        return lambda x:1
    if name=="Legendre":
        if n==2:
            r= lambda x:3*x**2-1
        if n==1:
            r = lambda x:x 
    if name=="Hermite":
        if n==2:
            r = lambda x:x**2-1
        if n==1:
            r = lambda x:x 
    if name=="Chebyshev":
        if n==2:
            r = lambda x:2*x**2-1
        if n==1:
            r = lambda x:x 
    if name=="Laguerre":
        if n==2:
            r = lambda x:x**2-4*x+2
        if n==1:
            r = lambda x:1-x
    return r


def datasets(n_k=np.array([[1,1],
                         [1,2]]),
                  T=1000,
                  name="Legendre",
                  dist="normal",
                  #dist="exponential",
                  seed=0):

    max = np.max(n_k[:,1])
    T += max
    V,_ = n_k.shape
    np.random.seed(seed)

    if dist=="normal":
        u=np.random.normal(size=(T,1))
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
            func = polynomial(n=n,name=name)
            y *= func(u[l+k,0])
            #print(y)
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

    datasets()