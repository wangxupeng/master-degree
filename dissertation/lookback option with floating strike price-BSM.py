import numpy as np
from scipy.stats import norm

#s0=initial stock price
#sigma= volatility 
#smin= the minimum asset price achieved to date
r = 0.1
sigma = 0.3
T = 1
K = 90
result=[]
for s0 in range(100):
    a1=((r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    a2=a1-sigma*np.sqrt(T)
    a3=((-r+sigma**2/2)*T)/(sigma*np.sqrt(T))
    y1=-2*(r-sigma**2/2)/(sigma**2)
    call_option=s0*norm.cdf(a1)-s0*sigma**2/(2*r)*norm.cdf(-a1)-s0*np.exp(-r*T)*(norm.cdf(a2)
    -sigma**2/(2*r)*np.exp(y1)*norm.cdf(-a3))
    result.append(call_option)

result_np=np.array(result)
np.savetxt('re.txt', result, delimiter=' , ')   # array
