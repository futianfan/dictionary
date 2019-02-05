
import numpy as np
N = 5
d = 10
A = np.random.random((N,d))
u,s,v = np.linalg.svd(A, full_matrices = False)
print(u.shape)
print(v.shape)
s = np.diag(s)
#print(s)

u = np.mat(u)
v = np.mat(v)
s = np.mat(s)

#print(u * s * v - A)
#print(A)

