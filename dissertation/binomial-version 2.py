import numpy as np
import matplotlib.pyplot as plt
# model & option parameters
S0 = 50 # initial index level
T = 3/12 # call option maturity
r = 0.1 # constant short rate
sigma = 0.4 # constant volatility factor of diffusion
# time parameters
M = 40 # time steps
dt = T / M # length of time interval
a = np.exp(r * dt) # discount factor per time interval
# binomial parameters
u = np.exp(sigma * np.sqrt(dt)) # up-movement
d = 1 / u # down-movement
p = (a - d) / (u - d) # martingale probability
S0=50
axi_y=[]
axi_x=[]
z1=M
for i in range(0,M+1):
    z1=M-1
    for j in range(0,i+1):
        S = S0*u**(j)*d**(i-j)
        axi_y.append(S)
        axi_x.append(i)
x=axi_x
y=axi_y
colors = axi_x
plt.scatter(x,y,c=colors, alpha=0.5)
plt.xlabel("time period")
plt.ylabel("the price calculated by binomial tree")
plt.show()











