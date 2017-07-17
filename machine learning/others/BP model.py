#----------
# build the dataset
#----------
from pybrain.datasets import SupervisedDataSet
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
x1=[float(x) for x in range(-1000,4000)]    #换成float型是为了后期可以标准化为0-1  
x2=[float(x) for x in range(-2000,8000,2)]  
x3=[float(x) for x in range(-5000,10000,3)]  
num=len(x1)   
y=[]  
for i in range(num):  
    y.append(x1[i]**2+x2[i]**2+x3[i]**2)
x1=[(x-min(x1))/(max(x1)-min(x1)) for x in x1]  
x2=[(x-min(x2))/(max(x2)-min(x2)) for x in x2]  
x3=[(x-min(x3))/(max(x3)-min(x3)) for x in x3]  
y=[(x-min(y))/(max(y)-min(y)) for x in y]  
#将X和Y转化为array格式  
x=np.array([x1,x2,x3]).T  
y=np.array(y)  
xdim=x.shape[1]       #确定输入特征维度  
ydim=1                #确定输出的维度  
    

ds = SupervisedDataSet(xdim, ydim)
for i in range(num):  
    ds.addSample(x[i],y[i])  
#----------
# build the network
#----------
from pybrain.structure import *
from pybrain.tools.shortcuts import buildNetwork

net = buildNetwork(xdim,
                   100,
                   15,# number of hidden units
                   ydim,
                   hiddenclass = TanhLayer,
                   outclass = LinearLayer
                   )
#----------
# train
#----------
from pybrain.supervised.trainers import BackpropTrainer
trainer = BackpropTrainer(net, ds, verbose = True)
trainer.trainUntilConvergence(maxEpochs = 100)

#----------
# evaluate
#----------

# neural net approximation
x11=[float(x) for x in range(0,100)]
x22=[float(x) for x in range(100,200)]
x33=[float(x) for x in range(200,300)]
num=len(x11)   
yy=[]  
for i in range(num):  
    yy.append(x11[i]**2+x22[i]**2+x33[i]**2)
x11=[(x-min(x11))/(max(x11)-min(x11)) for x in x11]  
x22=[(x-min(x22))/(max(x22)-min(x22)) for x in x22]  
x33=[(x-min(x33))/(max(x33)-min(x33)) for x in x33]  
yy=[(x-min(yy))/(max(yy)-min(yy)) for x in yy]  
xx=np.array([x11,x22,x33]).T
yy=np.array(yy)
out=[]
for i in range(len(xx)):
    out.append(net.activate(xx[i])[0])  
plt.figure(figsize=(8,4))
plt.plot(xx, out, 'ro--', label='predict number')  
plt.plot(xx, yy, 'ko-', label='true number')  
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.show()
