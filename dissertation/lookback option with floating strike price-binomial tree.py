import numpy as np
# model & option parameters
S0 = 50 # initial index level
T = 3/12 # call option maturity
r = 0.1 # constant short rate
sigma = 0.4 # constant volatility factor of diffusion
# time parameters
M = 3 # time steps
dt = T / M # length of time interval
a =np.exp(r * dt) # discount factor per time interval
# binomial parameters
u = np.exp(sigma * np.sqrt(dt)) # up-movement
d = 1 / u # down-movement
p = (a - d) / (u - d) # martingale probability
S = {}

for i in range(0,M+1):
    for j in range(0,i+1):
        S[i,j]=[S0*u**(j)*d**(i-j)]
#------------------------------------------------------stock prices of Binomial tree 

for i in range(0,M):
    for j in range(0,i+1):
        if S[(i,j)]>S[i+1,j+1] and j!=0:
            S[(i+1,j+1)].append(S[(i,j)][0])
        if S[(i,j)]>S[(i+1,j)] and j!=0:
            S[(i+1,j)].append(S[(i,j)][0])
            S[(i+1,j)].append(max(S[(i,j)]))
        if j==0:
            S[(i,j)].append(S0)

#------------------------------------------------------stock prices through Binomial tree pathes
            
for i in range(0,M):
    for j in range(0,i+1):
        S[(i,j,"f")]=[]
#------------------------------------------------------create option price       

for ia in range(M-1,-1,-1):
    for ja in range(ia,-1,-1):
        if ia+1==M:
            if (ia-1,ja) not in S and (ia-1,ja-1) in S:
                option_price1=(max(S[ia,ja])-max(S[ia,ja])*d)*(1-p)*np.exp(-r*dt)
                S[ia,ja,"f"].append(option_price1)              
            if (ia-1,ja) in S and (ia-1,ja-1) in S:
                option_price2=(max(S[ia-1,ja])-min(S[ia,ja])*d)*(1-p)*np.exp(-r*dt)
                S[ia,ja,"f"].append(option_price2)
                option_price3=(min(S[ia,ja])-min(S[ia-1,ja-1]))*(1-p)*np.exp(-r*dt)
                S[ia,ja,"f"].append(option_price3)                
            if (ia-1,ja) in S and (ia-1,ja-1) not in S:
                option_price4=((max(S[ia,ja])-min(S[ia,ja])*u)*p+(max(S[ia,ja])-min(S[ia,ja])*d)*(1-p))*np.exp(-r*dt)
                S[ia,ja,"f"].append(option_price4)
        else:
            if (ia-1,ja,"f") not in S and (ia-1,ja-1,"f") in S:
                option_price5=(max(S[ia+1,ja+1,"f"])*p+max(S[ia+1,ja,"f"])*(1-p))*np.exp(-r*dt)
                S[ia,ja,"f"].append(option_price5)
            if (ia-1,ja) in S and (ia-1,ja-1) in S:
                option_price6=(min(S[ia+1,ja,"f"])*(1-p))*np.exp(-r*dt)
                S[ia,ja,"f"].append(option_price6)
                option_price7=(max(S[ia+1,ja,"f"])*(1-p))*np.exp(-r*dt)
                S[ia,ja,"f"].append(option_price7)
            if (ia-1,ja) in S and (ia-1,ja-1) not in S:
                option_price8=(min(S[ia+1,ja+1,"f"])*p+max(S[ia+1,ja,"f"])*(1-p))*np.exp(-r*dt)
                S[ia,ja,"f"].append(option_price8)
            if (ia-1,ja) not in S and (ia-1,ja-1) not in S:
                option_price9=(min(S[ia+1,ja+1,"f"])*p+max(S[ia+1,ja,"f"])*(1-p))*np.exp(-r*dt)
                S[ia,ja,"f"].append(option_price9)
