import random
import math
import numpy as np
import matplotlib.pyplot as plt

def gd_gradient(A, x, b):
    return 2.*(A.dot(x)-b).dot(A)/len(A)


def nagd_gradient(A, x1, x0, b, i, t):
    y = x1 + ((i - 1.) / (i + 2.)) * (x1 - x0)
    return y - t * gd_gradient(A, y, b)


def mini_batch_gradient(A, x, b, batch_size):
    sample = np.random.choice(len(A), batch_size, replace=False)
    A_sample = np.array([A[i] for i in sample])
    b_sample = np.array([b[i] for i in sample])
    return 2. * (A_sample.dot(x)-b_sample).dot(A_sample) / batch_size


def sgd_gradient(A, x, b):
    index = random.randrange(len(A))
    sampled = A[index]
    return 2. * (sampled.dot(x) - b[index]) * sampled

def error(A, x, x_star, b):
    return math.pow(np.linalg.norm(A.dot(x) - A.dot(x_star)), 2) / len(A)
