"""
compute the value, 1st derivative, second derivative (Hessian) of
a dissimilarity constraint function g(sum{ij in D}(d_ij*A*d_ij'))
where A is a diagnal matrix.

df/dA = f'(sum{ij in D} sqrt{trace((x_i-x_j)A(x_i-x_j)')})
    *0.5*(sum(ij in D) (1/sqrt{trace(d_ij*A*d_ij')})*(d_ij'*d_ij))

"""
import numpy as np


def d_constraint(X, D, a):
    n, d = X.shape
    sum_dist = 0
    sum_deri1 = np.zeros(d)
    sum_deri2 = np.zeros((d, d))

    for i in range(n):
        for j in range(i+1, n):
            if D[i, j] == 1:
                d_ij = X[i] - X[j]
                dist_ij, deri1_d_ij, deri2_d_ij = distance1(a, d_ij)
                sum_dist += dist_ij
                sum_deri1 += deri1_d_ij
                sum_deri2 += deri2_d_ij
    fD, fD_1st_d, fD_2nd_d = gf(sum_dist, sum_deri1, sum_deri2)
    return [fD, fD_1st_d, fD_2nd_d]


def gf(sum_dist, sum_deri1, sum_deri2):
    fD = np.log(sum_dist)
    fD_1st_d = sum_deri1/sum_dist
    fD_2nd_d = sum_deri2/sum_dist - np.outer(sum_deri1, sum_deri1)/sum_dist**2
    return [fD, fD_1st_d, fD_2nd_d]


def distance1(a, d_ij):
    fudge = 0.000001
    dist_ij = np.sqrt(np.dot(d_ij**2, a))  # distance between X[i] and X[j]
    deri1_d_ij = 0.5*(d_ij**2)/(dist_ij + (dist_ij==0)*fudge)  # in case of dist_ij==0, shift dist_ij by 0.000001.
    deri2_d_ij = -0.25*np.outer(d_ij**2, d_ij**2)/(dist_ij**3+(dist_ij==0)*fudge)  # the same as last one.
    return [dist_ij, deri1_d_ij, deri2_d_ij]


"""
checking code. the result should be 1.039, [.25, .25], [[-0.125, -0.125], [-0.125, -0.125]]

x = np.array([[1, 2], [3, 4]])
d = np.ones((2, 2)) - np.tril(np.ones((2, 2)))
a = np.array([1, 1])

re = d_constraint(x, d, a)
print(re)
"""






