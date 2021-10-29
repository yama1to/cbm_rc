from re import X
import numpy as np 
import matplotlib.pyplot as plt 



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
        return lambda x:0
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


def datasets(T=1000):
    Legendre = polynomial(n=2,name="Legendre")
    u = np.random.normal(0,1,(T,1))
    y = Legendre(u)
    plt.plot(u)
    plt.plot(y)
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