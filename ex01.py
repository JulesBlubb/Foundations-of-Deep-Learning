import numpy as np
import math as m

''' Aufgabe 2'''
# a) calculate eigenvalue decomposition
A = [[1.75, -(np.sqrt(3)/4)],
     [-(np.sqrt(3)/4), 1.25]]

# calculate eigenvalues & eigenvectors
# eigenvalues: [2, 1]
values, vectors = np.linalg.eig(A)

Q = vectors

# Diagonalmatrix
V = np.zeros((2,2), int)
np.fill_diagonal(V,values)

# recreate A
B = np.dot(np.dot(Q,V), np.transpose(Q))
print(B)


# b)
# if Q.T * Q = I then Q is orthonormal (normalized Eigenvectors)
print(np.transpose(Q).dot(Q))


# c)
# Q * V^-1 * Q.T = A^-1
print((Q.dot(np.linalg.inv(V))).dot(np.transpose(Q)))
print(np.linalg.inv(A))

''' Aufgabe 3'''
D = [[3, 4],
     [6, 13]]

# D = L.dot(U)
# A * x = b
b = [1,2]

# lower triangular matrix
L = [[3, 0],
     [6, 13]]

U = [[3, 4],
     [0, 13]]



''' Aufgabe 4'''

x_1 = [24, 3, 2, 31]
x_2 = [27, 20, 26, 21]
x_3 = [30, 21, 27, 5]
x_4 = [26, 28, 25, 14]

""""
def norm1(v):
        sum = 0
        for c in v:
            sum = sum + abs(c)
            norm1 = sum
        return norm1

print(norm1(x_1))

def norm2(v):
    sum = 0
    for c in v:
        sum = sum + np.square(abs(c))
        norm2 = np.sqrt(sum)
    return norm2

print(norm2(x_1))

"""

# we could substitute 8 with p and call with 1 for norm1 or 2 for norm2
def norm(v, p):
    sum = 0
    for c in v:
        sum = sum + abs(c)**p
        norm = sum**(1/p)
    return norm

print(norm(x_1, 1))

# Maximumsnorm, Betrag der betragsgrößten Komponente
def maxnorm(v):
    sum = 0
    for c in v:
        maxnorm = np.max(abs(c))
    return maxnorm

print(maxnorm(x_1))

# TODO: Draw set of points ||x||_i = 1 for i{1,2,8,infinite} and x ∈ R²



''' Aufgabe 5'''

# a) A in C umbenannt
C = [[3.5, np.sqrt(3)/2],
    [np.sqrt(3)/2, 2.5]]

def determinant(m):
    return m[0][0] * m[1][1] - m[0][1]* m[1][0]


def trace(m):
    return m[0][0] + m[1][1]

print(determinant(C))
print(trace(C))

# TODO: eigvalues = np.linalg.eig(C)?

# b)

def Q(a):
    return [[m.cos(a), -m.sin(a)],
            [m.sin(a), m.cos(a)]]

def A_prime(a):
    return np.dot((np.dot(Q(a),(C))),
                  (np.transpose(Q(a))))

a_1 = m.pi/12
print(A_prime(a_1))
print(trace(A_prime(a_1)))
print(determinant(A_prime(a_1)))
# TODO: Eigenvalues for A_prime

# c)
a_2 = m.pi/3
print(A_prime(a_2))

