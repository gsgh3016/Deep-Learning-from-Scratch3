if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

"""x=Variable(np.array([[1,2,3],[4,5,6]]))
y=F.reshape(x,(6,))
y.backward(retain_grad=True)
print(x.grad)


x=Variable(np.array([[1,2,3],[4,5,6]]))
z=F.transpose(x)
z.backward()
print(x.grad)

x=Variable(np.random.rand(2,3))
y=x.transpose()
y=x.T
print(y)"""

"""a,b,c,d=1,2,3,4
x=np.random.rand(a,b,c,d)
print(x)
y=x.transpose(1,0,3,2)
print(y)"""

x=Variable(np.random.rand(1,2,3))
print(x)
y=F.transpose(x,(2,1,0))
print(y)