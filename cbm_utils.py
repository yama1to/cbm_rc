import matplotlib.pyplot as plt

def plot1(Up,Us,Rs,Hx,Hp,Yp,Dp,show = 1,save=1,dir_name = "trashfigure",fig_name="fig1"):
    fig=plt.figure(figsize=(16, 8))
    Nr=6
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    ax.set_title("Up")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    ax.set_title("Us")
    ax.plot(Us)
    ax.plot(Rs,"r:")
    #ax.plot(R2s,"b:")

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    ax.set_title("Hx")
    ax.plot(Hx)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    ax.set_title("Hp")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,5)
    ax.cla()
    ax.set_title("Yp")
    ax.plot(Yp)
    #ax.plot(y)

    ax = fig.add_subplot(Nr,1,6)
    ax.cla()
    ax.set_title("Dp")
    ax.plot(Dp)

    if show:plt.show()
    if save:plt.savefig("./{}/{}".format(dir_name,fig_name))

def plot2(Up,Hp,Yp,Dp,show = 1,save=1,dir_name = "trashfigure",fig_name="fig1"):
    fig=plt.figure(figsize=(20, 12))
    Nr=4
    ax = fig.add_subplot(Nr,1,1)
    ax.cla()
    #ax.set_title("input")
    ax.plot(Up)

    ax = fig.add_subplot(Nr,1,2)
    ax.cla()
    #ax.set_title("decoded reservoir states")
    ax.plot(Hp)

    ax = fig.add_subplot(Nr,1,3)
    ax.cla()
    #ax.set_title("predictive output")
    #ax.plot(train_Y)
    ax.plot(Yp)

    ax = fig.add_subplot(Nr,1,4)
    ax.cla()
    #ax.set_title("desired output")
    ax.plot(Dp)
    if show :plt.show()
    if save:plt.savefig("./{}/{}".format(dir_name,fig_name))
def plot_MC(DC,delay,MC,show = 1,save=1,dir_name = "trashfigure",fig_name="mc1"):
    plt.plot(DC)
    plt.ylabel("determinant coefficient")
    plt.xlabel("Delay k")
    plt.ylim([0,1])
    plt.xlim([0,delay])
    plt.title('MC ~ %3.2lf' % MC, x=0.8, y=0.7)
    if save:plt.savefig("./{}/{}".format(dir_name,fig_name))
    if show :plt.show()
    
def plot_delay(DC,N,pred,target,):
    fig=plt.figure(figsize=(8,8 ))
    Nr=N
    start = 0
    for i in range(Nr):
        ax = fig.add_subplot(Nr,1,i+1)
        ax.cla()
        ax.set_title("DC = %2f,delay = %s" % (DC[i],str(i)))
        ax.plot(pred.T[i,i:])
        ax.plot(target.T[i,i:])
    plt.show()