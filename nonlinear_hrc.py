
import cbmrc6b 
import matplotlib.pyplot as plt

cbm = cbmrc6b 
cbm.config()
cbm.execute()

#1.入力と内部状態の図
def make_graph_input_and_reservoir():
    input = cbm.Up

    #reservoir = cbm.Hx          #r_x
    decoded_reservoir = cbm.Hp

    

    print(input.shape,reservoir.shape,decoded_reservoir.shape)
    plt.plot(input , decoded_reservoir[:,0])
    plt.show()
#2.入力の周期


make_graph_input_and_reservoir()



"""
やりたいこと
1. 入力と内部状態の図表
2.周期

"""