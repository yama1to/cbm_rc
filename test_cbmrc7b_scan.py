import numpy as np
import itertools

#id,ex,seed,display,NN,Nh,alpha_i,alpha_r,alpha_b,alpha0,alpha1,alpha2,beta_i,beta_r,beta_b,Temp,lambda0

def scan1():
    Alpha_r=np.linspace(0.1,1,10)
    Alpha2 =np.linspace(0.1,1,10)

    #Alpha_r=np.random.uniform(0.1,10)
    for seed in np.arange(2):
        id=0
        for alpha_r,alpha2 in itertools.product(Alpha_r,Alpha2):
            print("python cbmrc7.py ex=ex1 display=0 seed=%d id=%d alpha_r=%f alpha2=%f" % (seed,id,alpha_r,alpha2))
            id+=1

def scan2():
    for id in np.arange(1000):
        seed = np.random.randint(0,100)
        alpha_r = np.random.uniform(0.1,1)
        alpha2  = np.random.uniform(0.1,1)
        Temp    = np.random.uniform(0.1,2)
        print("python cbmrc7.py ex=ex2 display=0 seed=%d id=0 alpha_r=%f alpha2=%f Temp=%f" % (seed,alpha_r,alpha2,Temp))

def scan3():
    for id in np.arange(1000):
        seed = np.random.randint(0,100)
        alpha_r = np.random.uniform(0.1,1)
        alpha2  = np.random.uniform(0.1,1)
        #Temp    = np.random.uniform(0.1,2)
        print("python cbmrc7.py ex=ex3 display=0 seed=%d id=0 alpha_r=%f alpha2=%f Temp=%f" % (seed,alpha_r,alpha2,1.0))

def scan4():
    #Alpha_r=np.linspace(0.1,1,10)
    Alpha2 =np.linspace(0.1,1,41)

    for seed in np.arange(20):
        id=0
        for alpha2 in Alpha2:
            print("python cbmrc7.py ex=ex4 display=0 seed=%d id=%d alpha_r=%f alpha2=%f" % (seed,id,0.3,alpha2))
            id+=1

def scan5():
    Alpha_r=np.linspace(0.1,1,41)
    #Alpha2 =np.linspace(0.1,1,41)

    for seed in np.arange(20):
        id=0
        for alpha_r in Alpha_r:
            print("python cbmrc7.py ex=ex5 display=0 seed=%d id=%d alpha_r=%f alpha2=%f" % (seed,id,alpha_r,0.5))
            id+=1

def scan6():
    #Alpha_r=np.linspace(0.1,1,41)
    #Alpha2 =np.linspace(0.1,1,41)
    Temp = np.linspace(0.1,2,41)

    for seed in np.arange(20):
        id=0
        for temp in Temp:
            print("python cbmrc7.py ex=ex6 display=0 seed=%d id=%d alpha_r=%f alpha2=%f Temp=%f" % (seed,id,0.3,0.5,temp))
            id+=1

def scan7():
    #Alpha_r=np.linspace(0.1,1,41)
    #Alpha2 =np.linspace(0.1,1,41)
    #Temp = np.linspace(0.1,2,41)
    Alpha_b = np.linspace(0.1,1,41)
    for seed in np.arange(20):
        id=0
        for alpha_b in Alpha_b:
            print("python cbmrc7.py ex=ex7 display=0 seed=%d id=%d alpha_r=%f alpha2=%f Temp=%f alpha_b=%f" % (seed,id,0.3,0.5,1.0,alpha_b))
            id+=1

def scan8():
    for id in np.arange(10000):
        seed = np.random.randint(0,100)
        alpha_r = np.random.uniform(0.1,1)
        alpha_b = np.random.uniform(0.1,2)
        alpha2  = np.random.uniform(0.1,2)
        print("python cbmrc7b.py ex=ex8 display=0 seed=%d id=0 alpha_r=%f alpha2=%f alpha_b=%f" % (seed,alpha_r,alpha2,alpha_b))


#NOTE alpha_r=0.25, alpha2=1.5, alpha_b=1.5,オーバーフローを考慮しなかったときの最適値
#NOTE
alpha_r=0.21
alpha2=0.88
alpha_b=0.24
#NOTE

def scan9():
    for seed in np.arange(20):
        for alpha_r in np.linspace(0,1,41):
            print("python cbmrc7b.py ex=ex9b display=0 seed=%d id=%d alpha_r=%f alpha2=%f alpha_b=%f" % (seed,0,alpha_r,alpha2,alpha_b))

def scan10():
    for seed in np.arange(20):
        for alpha2 in np.linspace(0.0,1,41):
            print("python cbmrc7b.py ex=ex10b display=0 seed=%d id=%d alpha_r=%f alpha2=%f alpha_b=%f" % (seed,0,alpha_r,alpha2,alpha_b))

def scan11():
    for seed in np.arange(20):
        for alpha_b in np.linspace(0.0,1,41):
            print("python cbmrc7b.py ex=ex11b display=0 seed=%d id=%d alpha_r=%f alpha2=%f alpha_b=%f" % (seed,0,alpha_r,alpha2,alpha_b))

scan9()
scan10()
scan11()
