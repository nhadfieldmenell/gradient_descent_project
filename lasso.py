import numpy as np
import math

def mini_batch_gradient(A, x, b, batch_size, l):
    sample = np.random.choice(len(A), batch_size, replace=False)
    A_sample = np.array([A[i] for i in sample])
    b_sample = np.array([b[i] for i in sample])
    lasso_factor = l * np.array([np.sign(i) if np.sign(i) != 0 else 1 for i in x])
    return 2. * (A_sample.dot(x)-b_sample).dot(A_sample) / batch_size + lasso_factor

def error(A, x, x_star, b, l):
    return math.pow(np.linalg.norm(A.dot(x) - A.dot(x_star)), 2) / len(A) + l * np.linalg.norm(x, 1)

