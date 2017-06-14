import numpy as np
import matplotlib.pyplot as plt
epsilon = 100
S0 = 20
M = 1000 #this can prove that the less time intervel the jagged of the plot.(时间跨度越小,锯齿化越严重.M这里也代表行数.)
T = 2.0
sigma = 0.25
r = 0.05
dt = T / M
S = np.zeros((M+1, epsilon))
S[0] = S0
for t in range(1, M+1) :
    S[t] = S[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * np.random.standard_normal(epsilon))
#this is because the distribution of "lnSt-lnS0" is [(mu-sigma^2/2)T,sigma^2*T](用他们的分布推导出上面的公式,还原成伊藤引理.)

fig = plt.figure(figsize=(10, 7.5))
plt.plot(S[:, :10], lw=1.5)# the right hand side in S is row, and the left hand side is col.(S里面的slice 前面是行,后面是列)
plt.xlabel('Time')
plt.ylabel('Index Level')
plt.title('Stock Prices Paths')
plt.grid()
plt.show()
