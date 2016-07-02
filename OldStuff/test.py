#from cvxopt import matrix
import numpy as np

A = np.array([[1.0, 2.0], [4.0, 5.0]])
B = np.array([ [1.0, 2.0], [3.0, 4.0] ])

print(A)


print(B)

c = np.array([10, 10])

d= A.dot(c)

print(d)