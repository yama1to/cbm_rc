"""
https://physionet.org/content/santa-fe/1.0.0/

    The data are presented in text form and have been split into two sequential parts,
    b1.txt and b2.txt.Each line contains simultaneous samples of three parameters;
    the interval between samples in successive lines is 0.5 seconds. 
    The first column is the heart rate, the second is the chest volume (respiration force),
    and the third is the blood oxygen concentration (measured by ear oximetry). 
    The sampling frequency for each measurement is 2 Hz 
    (i.e., the time interval between measurements in successive rows is 0.5 seconds).

"""
from generate_data_sequence_approximation import generate_data
import matplotlib.pyplot as plt 
import numpy as np

def generate_data(data,delay,train_num = 900,test_num = 500,):
        global normalize
        right = train_num

        right2 = test_num

        normalize = np.max(data[:right+right2+delay]) - np.min(data[:right+right2+delay])

        train_input = data[:right].reshape((train_num,1))
        train_input = train_input / normalize
        train_target = data[delay:right + delay].reshape((train_num,1))
        train_target = train_target / normalize

        
        test_input = data[right:right + right2 ].reshape((test_num,1))
        test_input = test_input / normalize
        test_target = data[right + delay :right + right2 +delay].reshape((test_num,1))
        test_target = test_target / normalize

        return train_input,train_target,test_input,test_target


def generate_santafe(delay = [1,2,3,4,5],train_num = 900,test_num = 500,):
    #"""
    with open('santafeA.txt', 'r', encoding='UTF-8') as f:
        data = np.array(list(f)).astype(int)

    #"""
    with open('santafeA2.txt', 'r', encoding='UTF-8') as f:
        tmp =  np.array(list(f)).astype(int)
        #data = tmp
        data = np.hstack((data,tmp))
    
    if int == type(delay):
        train_input,train_target,test_input,test_target = generate_data(data,delay,train_num = train_num,test_num = test_num,)
    
    if list == type(delay):
        Ny = len(delay)
        train_target = np.zeros((train_num,Ny))
        test_target = np.zeros((test_num,Ny))

        for i in range(Ny):
            if i == 0:
                train_input,train_t,test_input,test_t = generate_data(data,delay[0],train_num = train_num,test_num = test_num,)
                train_target[:,0] = train_t[:,0]
                test_target[:,0] = test_t[:,0]
            else:
                _,train_t,_,test_t = generate_data(data,delay[i],train_num = train_num,test_num = test_num,)
                train_target[:,i] = train_t[:,0]
                test_target[:,i] = test_t[:,0]


    return train_input,train_target,test_input,test_target,normalize

if __name__ == "__main__":
    #train_input,train_target,test_input,test_target = generate_santafe(delay = 1)
    #print(test_input.shape,test_target.shape)
    train_num = 600
    test_num = 500
    train_input,train_target,test_input,test_target,n = generate_santafe(delay = [10],train_num = train_num,test_num = test_num,)
    print(test_input.shape,test_target.shape)
    r1 = list(range(train_num))
    r2 = list(range(train_num,train_num+test_num))
    
    plt.plot(r1,train_input*n)
    #plt.plot(r1,train_target[:]*n)
    #plt.plot(r2,test_target[:]*n)
    plt.plot(r2,test_input*n)
    
    plt.show()
    fig=plt.figure(figsize=(20, 12))
    Nr=4
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("train_input")
    ax.plot(train_input)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("train_target")
    ax.plot(train_target)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("test_input")
    ax.plot(test_input)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    ax.set_title("test_target")
    ax.plot(test_target)
    plt.show()