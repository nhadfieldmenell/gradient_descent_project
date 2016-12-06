import numpy as np
import math

# A is the weight
# b is the targets

def logistic(x):
    return 1 / (1 + np.exp(-x))

def mini_batch_gradient(A, x, b, batch_size):
    sample = np.random.choice(len(A), batch_size, replace=False)
    A_sample = np.array([A[i] for i in sample])
    b_sample = np.array([b[i] for i in sample])
    
    return (logistic(A_sample.dot(x)) - b_sample).T.dot(A_sample)

def error(A, x, x_star, b):
    return np.mean(-(b * np.log(logistic(A.dot(x))) + (1. - b) * (np.log(1. - logistic(A.dot(x))))))
