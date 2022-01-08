import numpy as np
from numpy.core.numeric import zeros_like

from tqdm import tqdm

global NN
NN = 2**1
Temp = 1

# affine変換
def affine(z, W, b):
    return np.dot(z, W) + b

# affine変換勾配
def affine_back(du, z, W, b):
    dz = np.dot(du, W.T)
    dW = np.dot(z.T, du)
    db = np.dot(np.ones(z.shape[0]).T, du)
    return dz, dW, db

# 活性化関数(ReLU)
def relu(u):
    return np.maximum(0, u)

# 活性化関数(ReLU)勾配
def relu_back(dz, u):
    return dz * np.where(u > 0, 1, 0)

# 活性化関数(softmax)
def softmax(u):
    max_u = np.max(u, axis=1, keepdims=True)
    exp_u = np.exp(u-max_u)
    return exp_u/np.sum(exp_u, axis=1, keepdims=True)

# 誤差(交差エントロピー）
def cross_entropy_error(y, t):
    return -np.sum(t * np.log(np.maximum(y,1e-7)))/y.shape[0]

# 誤差(交差エントロピー）＋活性化関数(softmax)勾配
def softmax_cross_entropy_error_back(y, t):
    return (y - t)/y.shape[0]
def fy(h):
    return np.tanh(h)

def fyi(h):
    return np.arctanh(h)

def p2s(theta,p):
    return np.heaviside( np.sin(np.pi*(2*theta-p)),1)
def decode(u_s,step):
    x,y = u_s.shape
    t = y//step
    #print(x,t)
    fallingTime = 0
    dec = np.zeros((x,t))
    for X in range(x):
        for i in range(t):
            R = 0
            for j in range(step):
                dt = j/step
                if R:
                    if u_s[X,i*step+j] == 0:
                        fallingTime = dt
                        break
                #print(i,j,step,i*step+j)
                if u_s[X,i*step+j] == 1:
                    R = 1
            dec[X,i] = 2*fallingTime - 1
    return dec

def cbm(us,Wi,b1,hs,hx,Temp,dt,):
    z = (2*us-1)@Wi + b1
    hsign = 1 - 2*hs
    dx = hsign*(1.0+np.exp(hsign*z/Temp))*dt
    hx = hx + dx
    hs = np.heaviside(hx+hs-1,0)
    hx = np.fmin(np.fmax(hx,0),1)
    return hs,hx,dx

def gradient(dx,Temp,hsign,z):
    print("hsign:",str(hsign.shape))
    print("z:",str(z.shape))
    p  = hsign*z/Temp
    return dx*p

def cbm_back(dx,u,hs,W,Temp):
    hsign = 1-2*hs

    z = 2*u-1
    dW = gradient(dx,Temp,hsign,z)
    z = 2*W
    du = gradient(dx,Temp,hsign,z)
    z = np.ones(hs.shape[0])
    db = gradient(dx,Temp,hsign,z)
    return du, dW, db

def learn(batch_size,nx_train, t_train, W1, b1, W2, b2, W3, b3, lr):
    dt = 1/NN

    hs1 = np.zeros((100))
    hs2 = np.zeros((50))
    hs3 = np.zeros((10))
    
    hx1 = np.zeros((100))
    hx2 = np.zeros((50))
    hx3 = np.zeros((10))
    

    m = 0
    y = np.zeros((nx_train.shape[0],10))
    rs =1
    count = 0
    hc = np.zeros((10))
    hs = np.zeros(10)
    for k in tqdm(range(0,NN*nx_train.shape[0])):
        idx = int(k/NN)
        x,t = nx_train[idx], t_train[idx]
        hs_prev = hs.copy()
        rs_prev = rs
        theta = np.mod(m/NN,1) # (0,1)
        rs = p2s(theta,0)# 参照クロック
        us = p2s(theta,x)

        # 順伝播
        hs1,hx1,dx1 = cbm(us,W1,b1,hs1,hx1,Temp,dt,)
        z1 = hs1
        hs2,hx2,dx2 = cbm(z1,W2,b2,hs2,hx2,Temp,dt,)
        z2 = hs2
        hs3,hx3,dx3 = cbm(z2,W3,b3,hs3,hx3,Temp,dt,)
        z3 = softmax(hs3[:,np.newaxis])
 
        hs = z3[:,0]
        hc[(hs_prev == 1)& (hs==0)] = count
    
        # ref.clockの立ち上がり
        if rs_prev==0 and rs==1:
            yp = 2*hc/NN-1 # デコード、カウンタの値を連続値に変換
            hc = np.zeros(10) #カウンタをリセット
            #ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト
            y[m]=yp
            count = 0
            m += 1

            #back propagation
            dy = softmax_cross_entropy_error_back(y, t)
            du2, dW3, db3 = cbm_back(dx3,dy ,hs3, W3, b3) #cbm_back(dx,u,W,hsign,Temp)
            du1, dW2, db2 = cbm_back(dx2,du2,hs2, W2, b2)
            _ , dW1, db1 = cbm_back(dx1,du1,hs1,W1, b1)
            # du1, dW1, db1 = cbm_back(dx1,dy,hs1,W1, b1)
            # du2, dW2, db2 = cbm_back(dx2,du1,hs2, W2, b2)
            # _, dW3, db3 = cbm_back(dx3,du2 ,hs3, W3, b3) #cbm_back(dx,u,W,hsign,Temp)
           
            # 重み、バイアスの更新
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
            W3 = W3 - lr * dW3
            b3 = b3 - lr * db3

        #境界条件
        if k == (NN*nx_train.shape[0]-2):
            yp = 2*hc/NN-1 # デコード、カウンタの値を連続値に変換
            y[m]=yp

            #back propagation 

            dy = softmax_cross_entropy_error_back(y, t)
            
            du2, dW3, db3 = cbm_back(dx3,dy ,hs3, W3, b3) #cbm_back(dx,u,W,hsign,Temp)
            print(dx2.shape,du2.shape,hs2.shape, W2.shape)#(50,) (50, 10) (50,) (100, 50)
            du1, dW2, db2 = cbm_back(dx2,du2,hs2, W2, b2)
            _   , dW1, db1 = cbm_back(dx1,du1,hs1,W1, b1)
            # du1, dW1, db1 = cbm_back(dx1,dy,hs1,W1, b1)
            # du2, dW2, db2 = cbm_back(dx2,du1,hs2, W2, b2)
            # _, dW3, db3 = cbm_back(dx3,du2 ,hs3, W3, b3) #cbm_back(dx,u,W,hsign,Temp)

            # 重み、バイアスの更新
            W1 = W1 - lr * dW1
            b1 = b1 - lr * db1
            W2 = W2 - lr * dW2
            b2 = b2 - lr * db2
            W3 = W3 - lr * dW3
            b3 = b3 - lr * db3
        count += 1
    return W1, b1, W2, b2, W3, b3

def predict(x, W1, b1, W2, b2, W3, b3):
    dt = 1/NN

    hs1 = np.zeros((100))
    hs2 = np.zeros((50))
    hs3 = np.zeros((10))
    
    hx1 = np.zeros((100))
    hx2 = np.zeros((50))
    hx3 = np.zeros((10))
    
    Z = np.zeros((100*NN,10))
    m = 0
    y = np.zeros((x.shape[0],10))
    rs =1
    count = 0
    hc = np.zeros((10))
    hs = np.zeros(10)

    for k in tqdm(range(0,NN*x.shape[0])):
        idx = int(k/NN)
        x,t = nx_train[idx], t_train[idx]
        hs_prev = hs.copy()
        rs_prev = rs
        theta = np.mod(m/NN,1) # (0,1)
        rs = p2s(theta,0)# 参照クロック
        us = p2s(theta,x)
        # 順伝播
        #print(us.shape,W1.shape)
        u1,hx1,_ = cbm(us,W1,b1,hs1,hx1,Temp,dt,)
        z1 = u1
        u2,hx2,_ = cbm(z1,W2,b2,hs2,hx2,Temp,dt,)
        z2 = u2
        u3,hx3,_ = cbm(z2,W3,b3,hs3,hx3,Temp,dt,)
        z3 = softmax(u3[:,np.newaxis])
        #print(z3.shape)
        hs = z3[:,0]
        #print(hc.shape)
        hc[(hs_prev == 1)& (hs==0)] = count
    
        # ref.clockの立ち上がり

        if rs_prev==0 and rs==1:
            yp = 2*hc/NN-1 # デコード、カウンタの値を連続値に変換
            hc = np.zeros(10) #カウンタをリセット
            #ht = 2*hs-1 #リファレンスクロック同期用ラッチ動作をコメントアウト
            y[m]=yp
            count = 0
            m += 1
        #境界条件
        if k == (NN*x.shape[0]-1):
            yp = 2*hc/NN-1 # デコード、カウンタの値を連続値に変換
            y[m]=yp
        count += 1
    return y

    


import gzip
import numpy as np
# MNIST読み込み
def load_mnist( mnist_path ) :
    return _load_image(mnist_path + 'train-images-idx3-ubyte.gz'), \
           _load_label(mnist_path + 'train-labels-idx1-ubyte.gz'), \
           _load_image(mnist_path + 't10k-images-idx3-ubyte.gz'), \
           _load_label(mnist_path + 't10k-labels-idx1-ubyte.gz')

def _load_image( image_path ) :
    # 画像データの読み込み
    with gzip.open(image_path, 'rb') as f:
        buffer = f.read()
    size = np.frombuffer(buffer, np.dtype('>i4'), 1, offset=4)
    rows = np.frombuffer(buffer, np.dtype('>i4'), 1, offset=8)
    columns = np.frombuffer(buffer, np.dtype('>i4'), 1, offset=12)
    data = np.frombuffer(buffer, np.uint8, offset=16)
    image = np.reshape(data, (size[0], rows[0]*columns[0]))
    image = image.astype(np.float32)
    return image

def _load_label( label_path ) :
    # 正解データ読み込み
    with gzip.open(label_path, 'rb') as f:
        buffer = f.read()
    size = np.frombuffer(buffer, np.dtype('>i4'), 1, offset=4)
    data = np.frombuffer(buffer, np.uint8, offset=8)
    label = np.zeros((size[0], 10))
    for i in range(size[0]):
        label[i, data[i]] = 1
    return label

# 正解率
def accuracy_rate(y, t):
    print(y)
    print(t)
    max_y = np.argmax(y, axis=1)
    max_t = np.argmax(t, axis=1)
    return np.sum(max_y == max_t)/y.shape[0]


if __name__ == "__main__":

    # MNISTデータ読み込み
    x_train, t_train, x_test, t_test = load_mnist('/home/yamato/Downloads/cbm_rc/mnist_data/')

    # 入力データの正規化(0～1)
    nx_train = x_train/255
    nx_test  = x_test/255
    #print(x_train.shape, t_train.shape, x_test.shape, t_test.shape )
    # = (60000, 784) (60000, 10) (10000, 784) (10000, 10)
    #print(x_train.shape, t_train.shape, x_test.shape, t_test.shape )
    # ノード数設定
    d0 = nx_train.shape[1]
    d1 = 100 # 1層目のノード数
    d2 = 50  # 2層目のノード数
    d3 = 10
    # 重みの初期化(-0.1～0.1の乱数)
    np.random.seed(8)
    W1 = np.random.rand(d0, d1) * 0.2 - 0.1
    W2 = np.random.rand(d1, d2) * 0.2 - 0.1
    W3 = np.random.rand(d2, d3) * 0.2 - 0.1
    # バイアスの初期化(0)
    b1 = np.zeros(d1)
    b2 = np.zeros(d2)
    b3 = np.zeros(d3)

    # 学習率
    lr = 0.5
    # バッチサイズ
    batch_size =100
    # 学習回数
    epoch = 50
    
    # 予測（学習データ）
    y_train = predict(nx_train, W1, b1, W2, b2, W3, b3)
    # 予測（テストデータ）
    y_test = predict(nx_test, W1, b1, W2, b2, W3, b3)
    # 正解率、誤差表示
    #print(x_train.shape, t_train.shape, x_test.shape, t_test.shape )
    # = (60000, 784) (60000, 10) (10000, 784) (10000, 10)
    #print(y_train.shape, t_train.shape, y_test.shape, t_test.shape )
    # #(60000, 10) (60000, 10) (60000, 10) (10000, 10)
    train_rate, train_err = accuracy_rate(y_train, t_train), cross_entropy_error(y_train, t_train)
    test_rate, test_err = accuracy_rate(y_test, t_test), cross_entropy_error(y_test, t_test)
    print("{0:3d} train_rate={1:6.2f}% test_rate={2:6.2f}% train_err={3:8.5f} test_err={4:8.5f}".format((0), train_rate*100, test_rate*100, train_err, test_err))

    for i in range(epoch):
        # 学習
        W1, b1, W2, b2, W3, b3 = learn(batch_size,nx_train, t_train, W1, b1, W2, b2, W3, b3, lr)
        
        # 予測（学習データ）
        y_train = predict(nx_train, W1, b1, W2, b2, W3, b3)
        # 予測（テストデータ）
        y_test = predict(nx_test, W1, b1, W2, b2, W3, b3)
        # 正解率、誤差表示
        
        train_rate, train_err = accuracy_rate(y_train, t_train), cross_entropy_error(y_train, t_train)
        test_rate, test_err = accuracy_rate(y_test, t_test), cross_entropy_error(y_test, t_test)
        print("{0:3d} train_rate={1:6.2f}% test_rate={2:6.2f}% train_err={3:8.5f} test_err={4:8.5f}".format((i+1), train_rate*100, test_rate*100, train_err, test_err))