#!/usr/bin/python
# Copyright (c) 2014-2017 Katori Lab. All Rights Reserved
# NOTE: matplot minimal example

import numpy as np
import matplotlib.pyplot as plt
from generate_data_sequence import *

D, U = generate_simple_sinusoidal(500)
#D, U = generate_complex_sinusoidal(500)
#D, U = generate_coupled_lorentz(1000)

plt.subplot(2, 1, 1)
plt.plot(U)
plt.ylabel('U')

plt.subplot(2, 1, 2)
plt.plot(D)
plt.ylabel('D')
plt.show()
# plt.savefig("test.png")
