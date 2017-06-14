from math import *
# model & option parameters
S0 = 50 # initial index level
T = 3/12 # call option maturity
r = 0.1 # constant short rate
sigma = 0.4 # constant volatility factor of diffusion
# time parameters
M = 3 # time steps
dt = T / M # length of time interval
a =exp(r * dt) # discount factor per time interval
# binomial parameters
u = exp(sigma * sqrt(dt)) # up-movement
d = 1 / u # down-movement
p = (a - d) / (u - d) # martingale probability
S = {}
z1=M
for i in range(0,M+1):
    z1=M-1
    for j in range(0,i+1):
        S[i,j]=[S0*u**(j)*d**(i-j)]
for i in range(0,M):
    z1=M-1
    for j in range(0,i+1):
        if S[(i,j)]>S[i+1,j+1] and j>0:
            S[(i+1,j+1)].append(S[(i,j)][0])
        if S[(i,j)]>S[(i+1,j)] and j>0:
            S[(i+1,j)].append(S[(i,j)][0])
            S[(i+1,j)].append(max(S[(i,j)]))
        if j==0:
            S[(i,j)].append(S0)
for i in range(0,M):
    z1=M-1
    for j in range(0,i+1):
        S[(i,j,"f")]=[]
for ia in range(M,-1,-1):
    z1=M-1
    for ja in range(ia-1,-1,-1):
        if ia==M:
            for iaa in S[ia-1,ja]:
                result1=iaa-min(S[ia,ja])
                fresult=(0*p+result1*(1-p))*e**(-r*0.08333)
                fresult2=max(S[ia-1,ja])-min(S[ia-1,ja])
                S[ia-1,ja,"f"].append(fresult)
                S[ia-1,ja,"f"].append(fresult2)
        else:
            if min(S[ia-1,ja])*u<max(S[ia,ja+1]):
                result11=((min(S[ia,ja+1,"f"])*p+max(S[ia,ja,"f"])*(1-p))*e**(-r*0.08333))
                S[ia-1,ja,"f"].append(result11)
            if min(S[ia-1,ja])*u==max(S[ia,ja+1]):
                result22 = ((max(S[ia,ja+1,"f"])*p+max(S[ia,ja,"f"])*(1-p))*e**(-r*0.08333))
                S[ia-1,ja,"f"].append(result22)
print(S)
print(S[0,0,"f"])



