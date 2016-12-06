import random
import math
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import lsr_gradient as lsr
import lasso
import logistic
from sklearn import datasets
from collections import defaultdict
import seaborn as sns

def gd(A, x0, b, rate, iters, x_star, pkg):
    x = x0.copy()
    errors = []
    for i in range(iters):
        x -= rate * pkg.gd_gradient(A, x, b)
        errors.append(pkg.error(A, x, x_star))
    return errors

def momentum(A, x0, b, x_star, batch_size, iters, params, pkg, **kwargs):
    x = x0.copy()
    v = 0.
    errors = []
    rate = params['rate']
    gamma = params['gamma']
    for i in range(iters):
        v = gamma * v + rate * pkg.mini_batch_gradient(A, x, b, batch_size, **kwargs)
        x -= v
        errors.append(pkg.error(A, x, x_star, b, **kwargs))
    return errors


def sgd_mini_batch(A, x0, b, x_star, batch_size, iters, params, pkg, **kwargs):
    x = x0.copy()
    rate = params['rate']
    errors = []
    for i in range(iters):
        x -= rate * pkg.mini_batch_gradient(A, x, b, batch_size, **kwargs)
        errors.append(pkg.error(A, x, x_star, b, **kwargs))
    return errors

def sgd(A, x0, b, rate, iters, x_star, pkg):
    x = x0.copy()
    errors = []
    for i in range(iters):
        x -= rate * pkg.sgd_gradient(A, x, b)
        errors.append(pkg.error(A, x, x_star, b))
    return errors


def nesterov(A, x0, b, x_star, batch_size, iters, params, pkg, **kwargs):
    x = x0.copy()
    v = 0.
    errors = []
    rate = params['rate']
    gamma = params['gamma']
    for i in range(iters):
        v = gamma * v + rate * pkg.mini_batch_gradient(A, x-gamma*v, b, batch_size, **kwargs)
        x -= v
        errors.append(pkg.error(A, x, x_star, b, **kwargs))
    return errors


def adagrad(A, x0, b, x_star, batch_size, iters, params, pkg, **kwargs):
    x = x0.copy()
    G = [0. for i in range(len(x0))]
    errors = []
    rate = params['rate']
    epsilon = params['epsilon']
    for i in range(iters):
        grad = pkg.mini_batch_gradient(A, x, b, batch_size, **kwargs)
        G += grad ** 2
        adjusted_grad = grad / (np.sqrt(epsilon + G))
        x -= rate * adjusted_grad
        errors.append(pkg.error(A, x, x_star, b, **kwargs))
    return errors


def adadelta(A, x0, b, x_star, batch_size, iters, params, pkg, **kwargs):
    #Note that adadelta is very slow at the beginning and better for large, complex networks
    #There is no learning rate parameter, interestingly
    x = x0.copy()
    Eg2 = np.zeros(len(x0))
    errors = []
    Edt2 = np.zeros(len(x0))
    gamma = params['gamma']
    epsilon = params['epsilon']
    for i in range(iters):
        RMSdt_prev = np.sqrt(Edt2 + epsilon)
        grad = pkg.mini_batch_gradient(A, x, b, batch_size, **kwargs)
        Eg2 = gamma * Eg2 + (1. - gamma) * grad ** 2
        RMSg = np.sqrt(Eg2 + epsilon)
        dt = 0. - (RMSdt_prev / RMSg) * grad
        Edt2 = gamma * Edt2 + (1. - gamma) * dt ** 2
        x = x + dt
        errors.append(pkg.error(A, x, x_star, b, **kwargs))
    return errors


def RMSprop(A, x0, b, x_star, batch_size, iters, params, pkg, **kwargs):
    x = x0.copy()
    Eg2 = np.zeros(len(x0))
    errors = []
    rate = params['rate']
    gamma = params['gamma']
    epsilon = params['epsilon']
    for i in range(iters):
        grad = pkg.mini_batch_gradient(A, x, b, batch_size, **kwargs)
        Eg2 = gamma * Eg2 + (1. - gamma) * grad ** 2
        RMSg = np.sqrt(Eg2 + epsilon)
        x = x - (rate / RMSg) * grad
        errors.append(pkg.error(A, x, x_star, b, **kwargs))
    return errors


def adam(A, x0, b, x_star, batch_size, iters, params, pkg, **kwargs):
    x = x0.copy()
    m = np.zeros(len(x0))
    v = np.zeros(len(x0))
    errors = []
    rate = params['rate']
    epsilon = params['epsilon']
    B1 = params['B1']
    B2 = params['B2']
    for i in range(iters):
        grad = pkg.mini_batch_gradient(A, x, b, batch_size, **kwargs)
        m = B1 * m + (1. - B1) * grad
        v = B2 * v + (1. - B2) * grad ** 2
        m_hat = m / (1. - B1)
        v_hat = v / (1. - B2)
        x = x - (rate / (np.sqrt(v_hat) - epsilon)) * m_hat
        errors.append(pkg.error(A, x, x_star, b, **kwargs))
    return errors
        
def hypertune_sgd(A, x0, b, iters, batch_size, x_star, pkg, rate_range, trials):
    rates = []
    values = []
    for _ in range(trials):
        rate = random.uniform(rate_range[0], rate_range[1])
        rates.append(rate)
        values.append(sgd_mini_batch(A, x0, b, iters, rate, batch_size, x_star, pkg)[-1])
    return rates[np.argmin(values)]

def hypertune_two_param(A, x0, b, iters, batch_size, x_star, pkg, gamma_range, rate_range, trials, fn):
    rates = []
    gammas = []
    values = []
    for _ in range(trials):
        rate = random.uniform(rate_range[0], rate_range[1])
        rates.append(rate)
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        gammas.append(gamma)
        values.append(fn(A, x0, b, iters, gamma, rate, batch_size, x_star, pkg)[-1])
        print "%.4f, %.4f: %.9f" % (rate, gamma, values[-1])
    best = np.argmin(values)
    return gammas[best], rates[best]

def hypertune_rmsprop(A, x0, b, iters, batch_size, x_star, pkg, rate_range, gamma_range, trials):
    rates = []
    gammas = []
    values = []
    epsilon = 1e-8
    for _ in range(trials):
        rate = random.uniform(rate_range[0], rate_range[1])
        rates.append(rate)
        gamma = random.uniform(gamma_range[0], gamma_range[1])
        gammas.append(gamma)
        values.append(RMSprop(A, x0, b, rate, gamma, epsilon, batch_size, iters, x_star, lsr)[-1])
    best = np.argmin(values)
    return rates[best], gammas[best]

def hypertune_adam(A, x0, b, iters, batch_size, x_star, pkg, rate_range, B1_range, B2_range, trials):
    rates = []
    B1s = []
    B2s = []
    values = []
    epsilon = 1e-8
    for _ in range(trials):
        rate = random.uniform(rate_range[0], rate_range[1])
        rates.append(rate)
        B1 = random.uniform(B1_range[0], B1_range[1])
        B1s.append(B1)
        B2 = random.uniform(B2_range[0], B2_range[1])
        B2s.append(B2)
        values.append(adam(A, x0, b, rate, epsilon, B1, B2, batch_size, iters, x_star, lsr)[-1])
    best = np.argmin(values)
    return rates[best], B1s[best], B2s[best]

def random_search_tuning(A, x0, b, x_star, batch_size, iters, param2range, trials, pkg, fn, fixed2value={}, **kwargs):
    values = []
    params = defaultdict(list)
    for _ in range(trials):
        param_list = {fixed: fixed2value[fixed] for fixed in fixed2value}
        for param in param2range:
            params[param].append(random.uniform(param2range[param][0], param2range[param][1]))
            param_list[param] = params[param][-1]
        values.append(fn(A, x0, b, x_star, batch_size, iters, param_list, pkg, **kwargs)[-1])
    best = np.argmin(values)
    best_params = {param: params[param][best] for param in param2range}
    for fixed in fixed2value:
        best_params[fixed] = fixed2value[fixed]
    return best_params




def hypertune_lsr():
    n = 2000
    d = 200
    A = np.random.randn(n, d)
    x_star = np.random.randn(d)
    x_star = x_star / np.linalg.norm(x_star)
    b = A.dot(x_star) + .001 * np.random.normal(size=n)
    x0 = np.random.randn(d)
    x0 = x0 / np.linalg.norm(x0)

    batch_size = 32 
    epsilon = 1e-8
    iters = 40
    trials = 100

    rate_range = (.0001, .15)
    sgd_params = random_search_tuning(A, x0, b, x_star, batch_size, iters, {'rate': rate_range}, trials, lsr, sgd_mini_batch)

    rate_range = (.0001, .05)
    gamma_range = (.1, .9999999)
    momentum_params = random_search_tuning(A, x0, b, x_star, batch_size, iters, {'rate': rate_range, 'gamma': gamma_range}, trials, lsr, momentum)

    rate_range = (.0001, .08)
    gamma_range = (.1, .9999999)
    nesterov_params = random_search_tuning(A, x0, b, x_star, batch_size, iters, {'rate': rate_range, 'gamma': gamma_range}, trials, lsr, nesterov)

    rate_range = (.00001, .15)
    adagrad_params = random_search_tuning(A, x0, b, x_star, batch_size, iters, {'rate': rate_range}, trials, lsr, adagrad, {'epsilon': epsilon})

    adadelta_params = random_search_tuning(A, x0, b, x_star, batch_size, iters, {'gamma': gamma_range}, trials, lsr, adadelta, {'epsilon': epsilon})

    rate_range = (.00001, .15)
    gamma_range = (.8, .99999)
    rmsprop_params = random_search_tuning(A, x0, b, x_star, batch_size, iters, {'rate': rate_range, 'gamma': gamma_range}, trials, lsr, RMSprop, {'epsilon': epsilon})

    rate_range = (.00001, .15)
    B1_range = (.8, .99999)
    B2_range = (.9, .99999)
    adam_params = random_search_tuning(A, x0, b, x_star, batch_size, iters, {'rate': rate_range, 'B1': B1_range, 'B2': B2_range}, trials, lsr, adam, {'epsilon': epsilon})

    iters = 100
    sgd_mini_batch_values = sgd_mini_batch(A, x0, b, x_star, batch_size, iters, sgd_params, lsr)
    momentum_values = momentum(A, x0, b, x_star, batch_size, iters, momentum_params, lsr)
    nesterov_values = nesterov(A, x0, b, x_star, batch_size, iters, nesterov_params, lsr)
    adagrad_values = adagrad(A, x0, b, x_star, batch_size, iters, adagrad_params, lsr)
    adadelta_values = adadelta(A, x0, b, x_star, batch_size, iters, adadelta_params, lsr)
    rmsprop_values = RMSprop(A, x0, b, x_star, batch_size, iters, rmsprop_params, lsr)
    adam_values = adam(A, x0, b, x_star, batch_size, iters, adam_params, lsr)
    sgd_mini_batch_line, = plt.plot(sgd_mini_batch_values, 'g-', label='sgd mini batch error')
    momentum_line, = plt.plot(momentum_values, 'b-', label='momentum error')
    nesterov_line, = plt.plot(nesterov_values, 'c-', label='nesterov error')
    adagrad_line, = plt.plot(adagrad_values, 'y-', label='adagrad error')
    adadelta_line, = plt.plot(adadelta_values, 'k-', label='adadelta error')
    rmsprop_line, = plt.plot(rmsprop_values, 'm-', label='rmsprop error')
    adam_line, = plt.plot(adam_values, 'r-', label='adam error')

    plt.legend(handles=[sgd_mini_batch_line, momentum_line, nesterov_line, adagrad_line, adadelta_line, rmsprop_line, adam_line])
    plt.show()




def plot_lsr():
    n = 2000
    d = 200
    A = np.random.randn(n, d)
    x_star = np.random.randn(d)
    x_star = x_star / np.linalg.norm(x_star)
    b = A.dot(x_star) + .001 * np.random.normal(size=n)
    x0 = np.random.randn(d)
    x0 = x0 / np.linalg.norm(x0)

    gamma = 0.9
    sgd_t = .005
    batch_size = 32 
    iters = 200
    epsilon = 1e-8
    B1 = .9
    B2 = .999

    sgd_mini_batch_values = sgd_mini_batch(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t}, lsr)
    momentum_values = momentum(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma}, lsr)
    nesterov_values = nesterov(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma}, lsr)
    adagrad_values = adagrad(A, x0, b, x_star, batch_size, iters, {'rate': .1, 'epsilon': epsilon}, lsr)
    adadelta_values = adadelta(A, x0, b, x_star, batch_size, iters, {'epsilon': epsilon, 'gamma': gamma}, lsr)
    rmsprop_values = RMSprop(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'epsilon': epsilon, 'gamma': gamma}, lsr)
    adam_values = adam(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'epsilon': epsilon, 'B1': B1, 'B2': B2}, lsr)
    sgd_mini_batch_line, = plt.plot(sgd_mini_batch_values, 'g-', label='sgd mini batch error')
    momentum_line, = plt.plot(momentum_values, 'b-', label='momentum error')
    nesterov_line, = plt.plot(nesterov_values, 'y-', label='nesterov error')
    adagrad_line, = plt.plot(adagrad_values, 'k-', label='adagrad error')
    adadelta_line, = plt.plot(adadelta_values, 'm-', label='adadelta error')
    rmsprop_line, = plt.plot(rmsprop_values, 'r-', label='RMSprop error')
    adam_line, = plt.plot(adam_values, 'c-', label='adam error')
    plt.legend(handles=[sgd_mini_batch_line, momentum_line, nesterov_line, adagrad_line, adadelta_line, rmsprop_line, adam_line])
    plt.show()


def plot_lasso():
    n = 2000
    d = 1000
    sparsity = .01
    A = np.random.randn(n, d)
    x_star = np.squeeze(sparse.rand(1, d, density=sparsity).dot(np.identity(d)))
    x_star = x_star / np.linalg.norm(x_star)
    b = A.dot(x_star) + .0001 * np.random.normal(size=n)
    x0 = np.random.randn(d)
    x0 = x0 / np.linalg.norm(x0)

    gamma = 0.9
    sgd_t = .005
    batch_size = 32 
    iters = 150
    epsilon = 1e-8
    B1 = .9
    B2 = .999

    l = 1.5

    sgd_mini_batch_values = sgd_mini_batch(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t}, lasso, l=l)
    momentum_values = momentum(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma}, lasso, l=l)
    nesterov_values = nesterov(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma}, lasso, l=l)
    adagrad_values = adagrad(A, x0, b, x_star, batch_size, iters, {'rate': .1, 'epsilon': epsilon}, lasso, l=l)
    adadelta_values = adadelta(A, x0, b, x_star, batch_size, iters, {'gamma': gamma, 'epsilon': epsilon}, lasso, l=l)
    rmsprop_values = RMSprop(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma, 'epsilon': epsilon}, lasso, l=l)
    adam_values = adam(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'epsilon': epsilon, 'B1': B1, 'B2': B2}, lasso, l=l)
    sgd_mini_batch_line, = plt.plot(sgd_mini_batch_values, 'g-', label='sgd mini batch error')
    momentum_line, = plt.plot(momentum_values, 'b-', label='momentum error')
    nesterov_line, = plt.plot(nesterov_values, 'y-', label='nesterov error')
    adagrad_line, = plt.plot(adagrad_values, 'k-', label='adagrad error')
    adadelta_line, = plt.plot(adadelta_values, 'm-', label='adadelta error')
    rmsprop_line, = plt.plot(rmsprop_values, 'r-', label='RMSprop error')
    adam_line, = plt.plot(adam_values, 'c-', label='adam error')
    plt.legend(handles=[sgd_mini_batch_line, momentum_line, nesterov_line, adagrad_line, adadelta_line, rmsprop_line, adam_line])
    plt.show()


def random_draw(p):
    if random.random() < p:
        return 1
    return 0


def plot_logistic():
    data = datasets.load_iris()
    A = data.data[:100, :2]
    b = np.logical_not(data.target[:100])
    x_star = None
    x0 = np.zeros(2)

    gamma = 0.9
    sgd_t = .002
    batch_size = 64 
    iters = 4000
    epsilon = 1e-8
    B1 = .9
    B2 = .999
    sgd_mini_batch_values = sgd_mini_batch(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t}, logistic)
    momentum_values = momentum(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma}, logistic)
    nesterov_values = nesterov(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma}, logistic)
    adagrad_values = adagrad(A, x0, b, x_star, batch_size, iters, {'rate': .1, 'epsilon': epsilon}, logistic)
    adadelta_values = adadelta(A, x0, b, x_star, batch_size, iters, {'gamma': gamma, 'epsilon': epsilon}, logistic)
    rmsprop_values = RMSprop(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma, 'epsilon': epsilon}, logistic)
    adam_values = adam(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'epsilon': epsilon, 'B1': B1, 'B2': B2}, logistic)
    sgd_mini_batch_line, = plt.plot(sgd_mini_batch_values, 'g-', label='sgd mini batch error')
    momentum_line, = plt.plot(momentum_values, 'b-', label='momentum error')
    nesterov_line, = plt.plot(nesterov_values, 'y-', label='nesterov error')
    adagrad_line, = plt.plot(adagrad_values, 'k-', label='adagrad error')
    adadelta_line, = plt.plot(adadelta_values, 'm-', label='adadelta error')
    rmsprop_line, = plt.plot(rmsprop_values, 'r-', label='RMSprop error')
    adam_line, = plt.plot(adam_values, 'c-', label='adam error')
    plt.legend(handles=[sgd_mini_batch_line, momentum_line, nesterov_line, adagrad_line, adadelta_line, rmsprop_line, adam_line])
    plt.show()

    exit(1)
    sgd_mini_batch_values = sgd_mini_batch(A, x0, b, iters, sgd_t, batch_size, x_star, logistic)
    adadelta_values = adadelta(A, x0, b, gamma, epsilon, batch_size, iters, x_star, logistic)
    sgd_mini_batch_line, = plt.plot(sgd_mini_batch_values, 'g-', label='sgd mini batch error')
    adadelta_line, = plt.plot(adadelta_values, 'm-', label='adadelta error')
    plt.show()

def main():
    hypertune_lsr()
    exit(1)
    plot_lasso()
    plot_lsr()
    plot_logistic()


if __name__ == '__main__':
    main()
