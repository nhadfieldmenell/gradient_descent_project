from __future__ import division
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
sns.set(style='ticks', palette='Set2')
import pandas as pd
import numpy as np
import math




def logistic_func(theta, x):
    return float(1) / (1 + math.e**(-x.dot(theta)))
def log_gradient(theta, x, y):
    first_calc = logistic_func(theta, x) - np.squeeze(y)
    final_calc = first_calc.T.dot(x)
    return final_calc
def cost_func(theta, x, y):
    log_func_v = logistic_func(theta,x)
    y = np.squeeze(y)
    step1 = y * np.log(log_func_v)
    step2 = (1-y) * np.log(1 - log_func_v)
    final = -step1 - step2
    return np.mean(final)
def grad_desc(theta_values, X, y, lr=.001, converge_change=.001, iters=3000):
    #normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    #setup cost iter
    cost_iter = []
    cost = cost_func(theta_values, X, y)
    cost_iter.append([0, cost])
    change_cost = 1
    i = 1
    while(change_cost > converge_change):
    #for j in range(iters):
        #print theta_values
        old_cost = cost
        theta_values = theta_values - (lr * log_gradient(theta_values, X, y))
        cost = cost_func(theta_values, X, y)
        cost_iter.append([i, cost])
        change_cost = old_cost - cost
        i+=1
    return theta_values, np.array(cost_iter)
def pred_values(theta, X, hard=True):
    #normalize
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    pred_prob = logistic_func(theta, X)
    pred_value = np.where(pred_prob >= .5, 1, 0)
    if hard:
        return pred_value
    return pred_prob


def main():
    data = datasets.load_iris()
    X = data.data[:100, :2]
    y = data.target[:100]
    X_full = data.data[:100, :]

    """
    setosa = plt.scatter(X[:50,0], X[:50,1], c='b')
    versicolor = plt.scatter(X[50:,0], X[50:,1], c='r')
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend((setosa, versicolor), ("Setosa", "Versicolor"))
    sns.despine()
    plt.show()
    """

    shape = X.shape[1]
    y_flip = np.logical_not(y) #flip Setosa to be 1 and Versicolor to zero to be consistent
    betas = np.zeros(shape)
    fitted_values, cost_iter = grad_desc(betas, X, y_flip)
    predicted_y = pred_values(fitted_values, X)
    print np.sum(y_flip == predicted_y)
    
    plt.plot(cost_iter[:,0], cost_iter[:,1])
    plt.ylabel("Cost")
    plt.xlabel("Iteration")
    sns.despine()
    plt.show()


if __name__ == '__main__':
    main()
