"""
solving constraint optimization problem using Newton-Raphson method.
"""
import numpy as np
from Distance_Learning.D_constraint import d_constraint


def Newton(X, S, D, C):
    n, d = X.shape
    a = np.ones(d)

    fudge = 0.000001
    threshold1 = 0.001
    reduction = 2

    # sum(d'Ad)=sum(trace(d'Ad))=sum(trace(dd'A))=trace(sum(dd'A))=trace(sum(dd')A)
    # sum(d_ij'a) = sum(d_ij')a where d_ij = [(di1-dj1)**2...(din-djn)**2]'
    s_sum = np.zeros(d)
    d_sum = np.zeros(d)
    for i in range(n):
        for j in range(i+1, n):
            d_ij = X[i] - X[j]
            if S[i, j] == 1:
                s_sum += d_ij**2
            elif D[i, j] == 1:
                d_sum += d_ij**2

    tt = 1
    error = 1
    while error > threshold1:
        fd0, fd_1st_d, fd_2nd_d = d_constraint(X, D, a)
        obj_initial = s_sum.dot(a) - C*fd0
        fs_1st_d = s_sum  # first derivative of the S constraint
        gradient = fs_1st_d - C*fd_1st_d  # the gradient of objective
        Hessain = -C*fd_2nd_d + fudge*np.eye(d)
        invHessian = np.linalg.inv(Hessain)
        step = np.dot(invHessian, gradient)

    # Newton-Raphson update
    # Search over optimal lambda
        lambda1 = 1
        t = 1
        a_previous = 0
        atemp = a - lambda1*step  # x[n+1] = x[n]-f(x[n])/df([xn])dx[n]
        atemp = np.maximum(atemp, 0.000001)  # keep a to be positive

        fdd0 = d_constraint(X, D, atemp)
        obj = s_sum.dot(atemp) - C*fdd0[0]  # the a update to be atemp, compare this to obj_initial
        obj_previous = obj * 1.1  # just to get the while loop start

        while obj < obj_previous:
            obj_previous = obj
            a_previous = atemp
            lambda1 /= reduction
            atemp = a - lambda1*step
            atemp = np.maximum(atemp, 0.000001)
            fdd0 = d_constraint(X, D, atemp)
            obj = s_sum.dot(atemp) - C*fdd0[0]
            t += 1

        a = a_previous
        error = abs((obj_previous - obj_initial)/obj_previous)
        tt += 1
    return a

"""
checking code。
"""

x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
d = np.zeros((4, 4))
d[0, 1] = 1
s = np.zeros((4, 4))
s[2, 3] = 1
a = np.array([1, 1])

re = Newton(x, s, d, 1)
print(re)

"""
draw the obj function, obj_initial = s_sum.dot(a) - C*fd0
"""

"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
s_sum = np.array([4, 4])
d_sum = np.array([4, 4])
a1 = np.linspace(0.0001, 0.4, 300)
x, y = np.meshgrid(a1, a1)
# z = x**2 + y**2
z = 4*(x+y)-0.5*np.log(np.sqrt(4*(x+y)))
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x, y, z)
plt.show()
"""