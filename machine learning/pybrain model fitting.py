#----------
# build the dataset
#----------
from pybrain.datasets import SupervisedDataSet
import numpy, math

xvalues = numpy.linspace(0,2 * math.pi, 1001)
yvalues = 5 * numpy.sin(xvalues)

ds = SupervisedDataSet(1, 1)
for x, y in zip(xvalues, yvalues):
    ds.addSample((x,), (y,))

#----------
# build the network
#----------
from pybrain.structure import *
from pybrain.tools.shortcuts import buildNetwork

net = buildNetwork(1,
                   100, # number of hidden units
                   1,
                   bias = True,
                   hiddenclass = SigmoidLayer,
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
import pylab
# neural net approximation
pylab.plot(xvalues,
           [ net.activate([x]) for x in xvalues ], linewidth = 2,
           color = 'blue', label = 'NN output')

# target function
pylab.plot(xvalues,
           yvalues, linewidth = 2, color = 'red', label = 'target')

pylab.grid()
pylab.legend()
pylab.show()
