import numpy as np 
import matplotlib.pyplot as plt 

def to2(value,fp,bits):
    
    int2 = np.zeros(fp)
    dec2 = np.zeros(bits-1)

    integer = abs(int(value))
    decimal = abs(value) - integer

    #整数の変換
    for i in range(fp):
        if integer >= 2**i:
            int2[i] = 1
        integer = integer % 2**i

    #小数点以下の変換
    for i in range(1,1+bits-fp):
        if decimal >= 2**(-i):
            dec2[i-1] = 1
        decimal = decimal % 2**(-i)

    #整数と小数点を結合する
    if value < 0:
        txt = "-" 
        #txt = "1"
    else:
        txt = "+"
        #txt = "0"
    for i in int2:
        txt += str(int(i))

    txt += "."

    for i in dec2:
        txt += str(int(i))
    #print(txt)
    return txt

def to10(value,fp,bits):

    value_st = value
    sum = 0
    for i in range(1,1+fp):
        # print(i)
        # print(value_st)
        x = int(value_st[i])
        if x:
            sum += 2**(x-1)

    np.set_printoptions(precision = 32,suppress = True,floatmode = "fixed")
    for i in range(2+fp,len(value_st)):
        #print(value_st)
        
        x = int(value_st[i])
        if x:
            sum += 2**-(i-1-fp)
            #print(-(i-1-fp))

    if float(value) < 0:
        txt = "-" 
        #txt = "1"
    else:
        txt = ""
    txt += str(sum)
    return float(txt)


def quantize(value,fp=1,bits=8):
    
    value = to10(to2(value,fp,bits),fp,bits)
    return value


if __name__=="__main__":
    #print(np.get_printoptions())
    np.set_printoptions(precision = 32,suppress = True,floatmode = "fixed")
    #print(np.get_printoptions())
    np.random.seed(seed=0)
    b = np.random.uniform(0,1,50)/2# + 2**(-5)
    error = []
    for i in range(50):
        d = quantize(b[i],1,8)
        error.append(abs(d-b[i]))
    
    plt.title("error")
    plt.scatter(b,error)
    plt.show()
    # a = to2(b,1,8)
    # c = to10(a,1,8)
    # print("quantize:",d)
    # print("参考",bin(int(b*100000)))
    # print("b:",b)
    # print("a:",a)
    # print("c:",c)