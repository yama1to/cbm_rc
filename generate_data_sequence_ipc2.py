from re import X
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as st
from scipy.special import factorial
from tqdm import tqdm

# http://solidstatephysics.blog.fc2.com/blog-entry-44.html.
def Legendre(x,degree):
    P = 0
    for i in range(int(np.floor(degree/2)+1)):
        tmp = (-1)**i * factorial(2*degree-2*i) * x**(degree-2*i)
        tmp /= 2**degree * factorial(degree-i) * factorial(i) * factorial(degree-2*i)
        P += tmp
    return P

#http://solidstatephysics.blog.fc2.com/blog-entry-29.html.
def Hermite(x,degree):
    P = 0
    
    for k in range(int(np.floor(degree/2)+1)):
        tmp = (2*x)**(degree-2*k) * (-1)**k 
        tmp /= factorial(degree-2*k)*factorial(k)
        P += tmp
    P *=factorial(degree)
    return P

# http://math-juken.com/kijutu/chebyshev/
def Chebyshev(x,degree):
    T = 0
    for k in range(1+int(np.floor(degree/2))):
        t = (2*x)**(degree-2*k)
        tmp = (-1)**k*degree*factorial(degree-k)*t
        tmp /= 2*(degree-k)*factorial(degree-2*k)*factorial(k)
        T+= tmp
    return T

#https://www.ne.jp/asahi/music/marinkyo/matematiko/polinomoj-de-laguerre.html.ja
def Laguerre(x,degree):
    P = 0
    for k in range(degree+1):
        tmp = (-1)**k * factorial(degree)**2 * x**k
        tmp /= factorial(degree-k) * factorial(k)**2
        P+=tmp
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
                  new=0,
                  save = 0):
    """
    k:遅延長
    n:多項式の次数
    name:使用する多項式
    dist:入力の分布
    seed:乱数のシード値
    """
    if not new:
        u = np.load("./ipc3_dir/input_"+str(T)+name+".npy")
        d = np.load("./ipc3_dir/target_"+str(T)+str(int(n))+name+".npy")
        return u,d
    np.random.seed(seed)

    if dist=="normal":
        u = np.random.normal(loc = 0,scale=0.3,size=(T,1))
    elif dist=="uniform":
        u = np.random.uniform(-1,1,(T,1))
    elif dist=="arcsine":
        u = st.arcsine.rvs(loc = -1,scale=2,size=(T,1))#-0.5
    elif dist=="exponential":
        u = st.expon.rvs(loc = -1,scale=0.22,size=(T,1))
    max = np.max(np.max(abs(u)))
    u = u/max/2


    delay = np.arange(k)  # 遅延長z 
    d = np.empty((T, len(delay)))
    
    # for t in range(T):
    #     for k in range(len(delay)):
    #         y = polynomial(name,u,n)

    #         d[t, k] = y[t-delay[k],0]  # 遅延系列
    # d = np.empty((T, len(delay)))
    for k in range(len(delay)):
        for t in range(T):
            y = polynomial(name,u,n)
            d[t, k] = y[t-delay[k],0]  # 遅延系列

    if save:
                                #degree,delay,poly
        np.save("./ipc3_dir/input_"+str(T)+name+".npy",arr=u)
        np.save("./ipc3_dir/target_"+str(T)+str(int(n))+name+".npy",arr=d)
    return u,d





if __name__=="__main__":
    name_list = ["Legendre","Hermite","Chebyshev","Laguerre"]
    dist_list = ["uniform","normal","arcsine","exponential"]

    for i in tqdm(range(1,10+1)):
        for j in range(4):
            set = j
            dist = dist_list[set]
            name = name_list[set]
            u,d = datasets(k=20,n = i,
                        T=10200, 
                        name=name,
                        dist=dist,
                        #dist="exponential",
                        seed=0,
                        new=1,
                        save = 1)
            #plt.plot(u)
            #plt.savefig("./all_fig/ipc_data_{0}-{1}-u.png".format(i,j))
            #plt.clf()
            #plt.plot(d)
            #plt.savefig("./all_fig/ipc_data_{0}-{1}-d.png".format(i,j))
            #plt.clf()
            # if i==2:
            #     print(i,j)
            #     plt.plot(u,label = "u")
            #     plt.plot(d,label="d")
            #     plt.legend()
            #     plt.show()


    # set = 0
    # name = name_list[set]
    # dist = dist_list[set]
    # u,d = datasets(k=20,n = 2,
    #             T=1000, 
    #             name=name,
    #             dist=dist,
    #             #dist="exponential",
    #             seed=0,
    #             new=1,
    #             save = 0)
    # plt.plot(u)
    # plt.show()

    # plt.plot(d[:,:])
    # plt.show()
    # print(u.shape,d.shape)
