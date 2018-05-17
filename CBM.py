import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import os

# data in "image" folder
os.chdir("/home/daichiyamamoto/cbm_rc/image")

os.getcwd()

# differential equation
def UpdateX(s, neuronInput, precision, temperature):
    sign = -1 if s else 1
    return sign * (1 + np.exp(sign * neuronInput / temperature))*precision;

def UpdateS(s, x):
    if x >= 1.0:
        return 1
    elif x <= 0.0:
        return 0
    else:
        return s

# parameter
alpha_r = 1.0
beta_r = 0.8

N = 10 # number of neurons

t0 = 0.0
t1 = 5.0
T = 10000
dt = (t1-t0)/T # time grid

temperature = 1

# initial condotion
x = np.random.uniform(0, 1, N)
s = np.random.randint(0, 1, N)

t = np.arange(start = t0, stop = t1, step = dt) # time axis

_x = np.zeros(N)
_s = np.zeros(N)
z = np.zeros(N)

X = np.zeros((T,N))
S = np.zeros((T,N))

### Wr
Wr0 = np.zeros(N * N)
nonzeros = N * N * beta_r
Wr0[0:int(nonzeros / 2)] = 1
Wr0[int(nonzeros / 2):int(nonzeros)] = -1
np.random.shuffle(Wr0)
Wr0 = Wr0.reshape((N, N))
v = scipy.linalg.eigvals(Wr0)
lambda_max = max(abs(v))
Wr = Wr0 / lambda_max * alpha_r
print("lamda_max",lambda_max)
print("Wr:")
print(Wr)

for k in range(0,T):
    X[k,:] = x
    S[k,:] = s
    for i in range(0,N):
        z[i] = Wr[i,:]@s

        _x[i] = x[i] + UpdateX(s[i], z[i], dt, temperature)
        _s[i] = UpdateS(s[i], x[i])

        if _x[i] >= 1.0:
            _x[i] = 1.0
        elif _x[i] <= 0.0:
            _x[i] = 0.0
    x=_x
    s=_s

# plot graph
plt.subplot(2, 1, 1)
for i in range(0,N):
    plt.plot(t, X[:,i])
plt.xlabel("t", fontsize = 12)
plt.ylabel("x(t)", fontsize = 12)

plt.subplot(2, 1, 2)
for i in range(0,N):
    plt.plot(t, S[:,i])
plt.xlabel("t", fontsize = 12)
plt.ylabel("s(t)", fontsize = 12)

plt.savefig('State.png')
