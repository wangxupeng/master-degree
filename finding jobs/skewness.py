import math
import numpy as np
import tushare as ts
data = ts.get_k_data('600000', ktype='5')
datashare=[]
for i in data['open']:
    datashare.append(i)
n=len(datashare)
niu=sum(datashare)/n #E(X)
datashare2=[]
for i in datashare:
    a=i**2
    datashare2.append(a)
niu2=sum(datashare2)/n #E(X^2)
datashare3=[]
for i in datashare:
    b=i**3
    datashare3.append(b)
niu3=sum(datashare3)/n #E(X^3)
sigma = math.sqrt(niu2 - niu*niu)# standard deviation
skew = (niu3 - 3*niu*sigma**2 - niu**3)/(sigma**3)
print(skew)
