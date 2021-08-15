import matplotlib.pyplot as plt


def plot1(u,u_s,r_x,r_decoded, output,target):
    #fig=plt.figure(figsize=(20, 12))
    fig=plt.figure(figsize=(10, 6))
    Nr=6
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("input")
    ax.plot(u)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("encoded input")
    ax.plot(u_s)
    #ax.plot(R2s,"b:")

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("internal state of neuron")
    ax.plot(r_x)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    ax.set_title("decoded state")
    ax.plot(r_decoded)

    ax = fig.add_subplot(Nr,1,5)
    ax.cla()
    ax.set_title("prediction")
    ax.plot(output)

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.set_title("target")
    ax.plot(target)

    plt.show()