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
import matplotlib.pyplot as plt 
import numpy as np

def generate_santafe(delay = 1):
    #"""
    with open('/home/yamato/Downloads/cbm_rc/santafeA.txt', 'r', encoding='UTF-8') as f:
        data = np.array(list(f)).astype(int)

    #"""
    with open('/home/yamato/Downloads/cbm_rc/santafeA2.txt', 'r', encoding='UTF-8') as f:
        tmp =  np.array(list(f)).astype(int)
        #data = tmp
        data = np.hstack((data,tmp))
    
    

    train_num = 900
    test_num = 100

    right = 900

    right2 = 100

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

if __name__ == "__main__":
    train_input,train_target,test_input,test_target = generate_santafe(delay = 1)

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