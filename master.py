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
import pdb

def momentum(A, x0, b, x_star, batch_size, iters, params, pkg):
    x = x0.copy()
    v = 0.
    errors = []
    rate = params['rate']
    gamma = params['gamma']
    if pkg == lasso:
        l = params['lambda']
    for i in range(iters):
        if pkg == lasso:
            v = gamma * v + rate * pkg.mini_batch_gradient(A, x, b, batch_size, l)
        else:
            v = gamma * v + rate * pkg.mini_batch_gradient(A, x, b, batch_size)
        x -= v
        if pkg == lasso:
            errors.append(pkg.error(A, x, x_star, b, l))
        else:
            errors.append(pkg.error(A, x, x_star, b))
    return errors


def sgd_mini_batch(A, x0, b, x_star, batch_size, iters, params, pkg, **kwargs):
    x = x0.copy()
    rate = params['rate']
    errors = []
    if pkg == lasso:
        l = params['lambda']
    for i in range(iters):
        if pkg == lasso:
            x -= rate * pkg.mini_batch_gradient(A, x, b, batch_size, l)
        else:
            x -= rate * pkg.mini_batch_gradient(A, x, b, batch_size)
        if pkg == lasso:
            errors.append(pkg.error(A, x, x_star, b, l))
        else:
            errors.append(pkg.error(A, x, x_star, b))
    return errors

def nesterov(A, x0, b, x_star, batch_size, iters, params, pkg):
    x = x0.copy()
    v = 0.
    errors = []
    rate = params['rate']
    gamma = params['gamma']
    if pkg == lasso:
        l = params['lambda']
    for i in range(iters):
        if pkg == lasso:
            v = gamma * v + rate * pkg.mini_batch_gradient(A, x-gamma*v, b, batch_size, l)
        else:
            v = gamma * v + rate * pkg.mini_batch_gradient(A, x-gamma*v, b, batch_size)
        x -= v
        if pkg == lasso:
            errors.append(pkg.error(A, x, x_star, b, l))
        else:
            errors.append(pkg.error(A, x, x_star, b))
    return errors


def adagrad(A, x0, b, x_star, batch_size, iters, params, pkg):
    x = x0.copy()
    G = [0. for i in range(len(x0))]
    errors = []
    rate = params['rate']
    epsilon = params['epsilon']
    if pkg == lasso:
        l = params['lambda']
    for i in range(iters):
        if pkg == lasso:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size, l)
        else:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size)
        G += grad ** 2
        adjusted_grad = grad / (np.sqrt(epsilon + G))
        x -= rate * adjusted_grad
        if pkg == lasso:
            errors.append(pkg.error(A, x, x_star, b, l))
        else:
            errors.append(pkg.error(A, x, x_star, b))
    return errors

def adagrad2(A, x0, b, x_star, batch_size, iters, params, pkg):
    x = x0.copy()
    G = np.zeros((len(x0), len(x0)))
    errors = []
    rate = params['rate']
    epsilon = np.array([params['epsilon'] for i in range(len(x0))])
    epsilon = np.diag(epsilon)
    print epsilon
    if pkg == lasso:
        l = params['lambda']
    for i in range(iters):
        if pkg == lasso:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size, l)
        else:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size)
        print '\n\n'
        print grad
        G += np.outer(grad, grad)
        print G
        adjusted_grad = (1 / (np.sqrt(G + epsilon))).dot(grad)
        print rate * adjusted_grad
        x -= rate * adjusted_grad
        print x
        if pkg == lasso:
            errors.append(pkg.error(A, x, x_star, b, l))
        else:
            errors.append(pkg.error(A, x, x_star, b))
    return errors

def adadelta(A, x0, b, x_star, batch_size, iters, params, pkg):
    #Note that adadelta is very slow at the beginning and better for large, complex networks
    #There is no learning rate parameter, interestingly
    x = x0.copy()
    Eg2 = np.zeros(len(x0))
    errors = []
    Edt2 = np.zeros(len(x0))
    gamma = params['gamma']
    epsilon = params['epsilon']
    if pkg == lasso:
        l = params['lambda']
    for i in range(iters):
        RMSdt_prev = np.sqrt(Edt2 + epsilon)
        if pkg == lasso:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size, l)
        else:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size)
        Eg2 = gamma * Eg2 + (1. - gamma) * grad ** 2
        RMSg = np.sqrt(Eg2 + epsilon)
        dt = 0. - (RMSdt_prev / RMSg) * grad
        Edt2 = gamma * Edt2 + (1. - gamma) * dt ** 2
        x = x + dt
        if pkg == lasso:
            errors.append(pkg.error(A, x, x_star, b, l))
        else:
            errors.append(pkg.error(A, x, x_star, b))
    return errors


def rmsprop(A, x0, b, x_star, batch_size, iters, params, pkg):
    x = x0.copy()
    Eg2 = np.zeros(len(x0))
    errors = []
    rate = params['rate']
    gamma = params['gamma']
    epsilon = params['epsilon']
    if pkg == lasso:
        l = params['lambda']
    for i in range(iters):
        if pkg == lasso:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size, l)
        else:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size)
        Eg2 = gamma * Eg2 + (1. - gamma) * grad ** 2
        RMSg = np.sqrt(Eg2 + epsilon)
        x = x - (rate / RMSg) * grad
        if pkg == lasso:
            errors.append(pkg.error(A, x, x_star, b, l))
        else:
            errors.append(pkg.error(A, x, x_star, b))
    return errors


def adam(A, x0, b, x_star, batch_size, iters, params, pkg):
    x = x0.copy()
    m = np.zeros(len(x0))
    v = np.zeros(len(x0))
    errors = []
    rate = params['rate']
    epsilon = params['epsilon']
    B1 = params['B1']
    B2 = params['B2']
    if pkg == lasso:
        l = params['lambda']
    for i in range(iters):
        if pkg == lasso:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size, l)
        else:
            grad = pkg.mini_batch_gradient(A, x, b, batch_size)
        m = B1 * m + (1. - B1) * grad
        v = B2 * v + (1. - B2) * grad ** 2
        m_hat = m / (1. - math.pow(B1, i+1))
        v_hat = v / (1. - math.pow(B2, i+1))
        x = x - (rate / (np.sqrt(v_hat) + epsilon)) * m_hat
        if pkg == lasso:
            errors.append(pkg.error(A, x, x_star, b, l))
        else:
            errors.append(pkg.error(A, x, x_star, b))
    return errors


def random_search_tuning(As, x0s, bs, x_stars, batch_size, iters, param2range, trials, pkg, fn, fixed2value={}):
    values = []
    params = defaultdict(list)
    for _ in range(trials):
        param_list = {fixed: fixed2value[fixed] for fixed in fixed2value}
        for param in param2range:
            params[param].append(random.uniform(param2range[param][0], param2range[param][1]))
            param_list[param] = params[param][-1]
        value = 0.
        for i in range(len(As)):
            value += fn(As[i], x0s[i], bs[i], x_stars[i], batch_size, iters, param_list, pkg)[-1]
        values.append(value / len(As))
    best = np.argmin(values)
    best_params = {param: params[param][best] for param in param2range}
    for fixed in fixed2value:
        best_params[fixed] = fixed2value[fixed]
    return best_params


def hypertune_parameters(data_generator, validation_count, batch_size, tune_iters, eval_iters, trials, pkg, method2params, method2fixed, title):
    #pdb.set_trace()
    As = []
    x0s = []
    bs = []
    x_stars = []
    for _ in range(validation_count):
        A, x0, b, x_star = data_generator()
        As.append(A)
        x0s.append(x0)
        bs.append(b)
        x_stars.append(x_star)
    sgd_params = random_search_tuning(As, x0s, bs, x_stars, batch_size, tune_iters, method2params['sgd'], trials, pkg, sgd_mini_batch, method2fixed['sgd'])
    print sgd_params
    momentum_params = random_search_tuning(As, x0s, bs, x_stars, batch_size, tune_iters, method2params['momentum'], trials, pkg, momentum, method2fixed['momentum'])
    print momentum_params
    nesterov_params = random_search_tuning(As, x0s, bs, x_stars, batch_size, tune_iters, method2params['nesterov'], trials, pkg, nesterov, method2fixed['nesterov'])
    print nesterov_params
    adagrad_params = random_search_tuning(As, x0s, bs, x_stars, batch_size, tune_iters, method2params['adagrad'], trials, pkg, adagrad, method2fixed['adagrad'])
    print adagrad_params
    adadelta_params = random_search_tuning(As, x0s, bs, x_stars, batch_size, tune_iters, method2params['adadelta'], trials, pkg, adadelta, method2fixed['adadelta'])
    print adadelta_params
    rmsprop_params = random_search_tuning(As, x0s, bs, x_stars, batch_size, tune_iters, method2params['rmsprop'], trials, pkg, rmsprop, method2fixed['rmsprop'])
    print rmsprop_params
    adam_params = random_search_tuning(As, x0s, bs, x_stars, batch_size, tune_iters, method2params['adam'], trials, pkg, adam, method2fixed['adam'])
    print adam_params

    A, x0, b, x_star = data_generator()

    sgd_mini_batch_values = sgd_mini_batch(A, x0, b, x_star, batch_size, eval_iters, sgd_params, pkg)
    momentum_values = momentum(A, x0, b, x_star, batch_size, eval_iters, momentum_params, pkg)
    nesterov_values = nesterov(A, x0, b, x_star, batch_size, eval_iters, nesterov_params, pkg)
    adagrad_values = adagrad(A, x0, b, x_star, batch_size, eval_iters, adagrad_params, pkg)
    adadelta_values = adadelta(A, x0, b, x_star, batch_size, eval_iters, adadelta_params, pkg)
    rmsprop_values = rmsprop(A, x0, b, x_star, batch_size, eval_iters, rmsprop_params, pkg)
    adam_values = adam(A, x0, b, x_star, batch_size, eval_iters, adam_params, pkg)
    sgd_mini_batch_line, = plt.plot(sgd_mini_batch_values, 'g-', label='sgd mini batch error')
    momentum_line, = plt.plot(momentum_values, 'b-', label='momentum error')
    nesterov_line, = plt.plot(nesterov_values, 'c-', label='nesterov error')
    adagrad_line, = plt.plot(adagrad_values, 'y-', label='adagrad error')
    adadelta_line, = plt.plot(adadelta_values, 'k-', label='adadelta error')
    rmsprop_line, = plt.plot(rmsprop_values, 'm-', label='rmsprop error')
    adam_line, = plt.plot(adam_values, 'r-', label='adam error')

    plt.xlabel('iterations')
    plt.ylabel('training cost')
    plt.title('%s with Hyperparameter Optimization' % title)
    plt.legend(handles=[sgd_mini_batch_line, momentum_line, nesterov_line, adagrad_line, adadelta_line, rmsprop_line, adam_line])
    plt.show()


def hypertune_lsr():
    batch_size = 32 
    epsilon = 1e-8
    tune_iters = 40
    eval_iters = 100
    trials = 100
    validation_count = 5

    method2params = defaultdict(set)
    method2fixed = defaultdict(set)
    rate_range = (.0001, .15)
    method2params['sgd'] = {'rate': rate_range}

    rate_range = (.0001, .05)
    gamma_range = (.1, .9999999)
    method2params['momentum'] = {'rate': rate_range, 'gamma': gamma_range}

    rate_range = (.0001, .08)
    gamma_range = (.1, .9999999)
    method2params['nesterov'] = {'rate': rate_range, 'gamma': gamma_range}

    rate_range = (.00001, .15)
    method2params['adagrad'] = {'rate': rate_range}
    method2fixed['adagrad'] = {'epsilon': epsilon}

    method2params['adadelta'] = {'gamma': gamma_range}
    method2fixed['adadelta'] = {'epsilon': epsilon}

    rate_range = (.00001, .15)
    gamma_range = (.8, .99999)
    method2params['rmsprop'] = {'rate': rate_range, 'gamma': gamma_range}
    method2fixed['rmsprop'] = {'epsilon': epsilon}

    rate_range = (.00001, .15)
    B1_range = (.8, .99999)
    B2_range = (.9, .99999)
    method2params['adam'] = {'rate': rate_range, 'B1': B1_range, 'B2': B2_range}
    method2fixed['adam'] = {'epsilon': epsilon}

    hypertune_parameters(generate_lsr_data, validation_count, batch_size, tune_iters, eval_iters, trials, lsr, method2params, method2fixed, 'LSR')


def hypertune_lasso():
    batch_size = 32 
    epsilon = 1e-8
    tune_iters = 30
    eval_iters = 100
    trials = 100
    validation_count = 5

    l = 1.5

    method2params = defaultdict(set)
    method2fixed = defaultdict(set)
    rate_range = (.0001, .15)
    method2params['sgd'] = {'rate': rate_range}
    method2fixed['sgd'] = {'lambda': l}

    rate_range = (.0001, .05)
    gamma_range = (.1, .9999999)
    method2params['momentum'] = {'rate': rate_range, 'gamma': gamma_range}
    method2fixed['momentum'] = {'lambda': l}

    rate_range = (.0001, .08)
    gamma_range = (.1, .9999999)
    method2params['nesterov'] = {'rate': rate_range, 'gamma': gamma_range}
    method2fixed['nesterov'] = {'lambda': l}

    rate_range = (.00001, .15)
    method2params['adagrad'] = {'rate': rate_range}
    method2fixed['adagrad'] = {'epsilon': epsilon, 'lambda': l}

    method2params['adadelta'] = {'gamma': gamma_range}
    method2fixed['adadelta'] = {'epsilon': epsilon, 'lambda': l}

    rate_range = (.00001, .15)
    gamma_range = (.8, .99999)
    method2params['rmsprop'] = {'rate': rate_range, 'gamma': gamma_range}
    method2fixed['rmsprop'] = {'epsilon': epsilon, 'lambda': l}

    rate_range = (.00001, .15)
    B1_range = (.8, .99999)
    B2_range = (.9, .99999)
    method2params['adam'] = {'rate': rate_range, 'B1': B1_range, 'B2': B2_range}
    method2fixed['adam'] = {'epsilon': epsilon, 'lambda': l}

    print method2params
    l = 1.5
    hypertune_parameters(generate_lasso_data, validation_count, batch_size, tune_iters, eval_iters, trials, lasso, method2params, method2fixed)

def hypertune_logistic():
    batch_size = 32 
    epsilon = 1e-8
    tune_iters = 100
    eval_iters = 2000
    trials = 50
    validation_count = 1

    method2params = defaultdict(set)
    method2fixed = defaultdict(set)
    rate_range = (.00001, .005)
    method2params['sgd'] = {'rate': rate_range}

    rate_range = (.00001, .005)
    gamma_range = (.1, .9999999)
    method2params['momentum'] = {'rate': rate_range, 'gamma': gamma_range}

    rate_range = (.00001, .005)
    gamma_range = (.1, .9999999)
    method2params['nesterov'] = {'rate': rate_range, 'gamma': gamma_range}

    rate_range = (.00001, .005)
    method2params['adagrad'] = {'rate': rate_range}
    method2fixed['adagrad'] = {'epsilon': epsilon}

    method2params['adadelta'] = {'gamma': gamma_range}
    method2fixed['adadelta'] = {'epsilon': epsilon}

    rate_range = (.00001, .005)
    gamma_range = (.8, .99999)
    method2params['rmsprop'] = {'rate': rate_range, 'gamma': gamma_range}
    method2fixed['rmsprop'] = {'epsilon': epsilon}

    rate_range = (.00001, .001)
    B1_range = (.8, .99999)
    B2_range = (.9, .99999)
    method2params['adam'] = {'rate': rate_range, 'B1': B1_range, 'B2': B2_range}
    method2fixed['adam'] = {'epsilon': epsilon}

    hypertune_parameters(generate_logistic_data, validation_count, batch_size, tune_iters, eval_iters, trials, logistic, method2params, method2fixed)


def generate_lsr_data():
    n = 2000
    d = 200
    A = np.random.randn(n, d)
    x_star = np.random.randn(d)
    x_star = x_star / np.linalg.norm(x_star)
    b = A.dot(x_star) + .001 * np.random.normal(size=n)
    x0 = np.random.randn(d)
    x0 = x0 / np.linalg.norm(x0)
    return A, x0, b, x_star


def plot_lsr():
    A, x0, b, x_star = generate_lsr_data()

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
    adagrad_values = adagrad(A, x0, b, x_star, batch_size, iters, {'rate': .01, 'epsilon': epsilon}, lsr)
    adadelta_values = adadelta(A, x0, b, x_star, batch_size, iters, {'epsilon': epsilon, 'gamma': gamma}, lsr)
    rmsprop_values = rmsprop(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'epsilon': epsilon, 'gamma': gamma}, lsr)
    adam_values = adam(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'epsilon': epsilon, 'B1': B1, 'B2': B2}, lsr)
    sgd_mini_batch_line, = plt.plot(sgd_mini_batch_values, 'g-', label='sgd mini batch error')
    momentum_line, = plt.plot(momentum_values, 'b-', label='momentum error')
    nesterov_line, = plt.plot(nesterov_values, 'y-', label='nesterov error')
    adagrad_line, = plt.plot(adagrad_values, 'k-', label='adagrad error')
    adadelta_line, = plt.plot(adadelta_values, 'm-', label='adadelta error')
    rmsprop_line, = plt.plot(rmsprop_values, 'r-', label='RMSprop error')
    adam_line, = plt.plot(adam_values, 'c-', label='adam error')
    plt.legend(handles=[sgd_mini_batch_line, momentum_line, nesterov_line, adagrad_line, adadelta_line, rmsprop_line, adam_line])
    plt.xlabel('iterations')
    plt.ylabel('training cost')
    plt.title('LSR with Standard Settings')
    plt.show()


def generate_lasso_data():
    n = 2000
    d = 1000
    sparsity = .01
    A = np.random.randn(n, d)
    x_star = np.squeeze(sparse.rand(1, d, density=sparsity).dot(np.identity(d)))
    x_star = x_star / np.linalg.norm(x_star)
    b = A.dot(x_star) + .0001 * np.random.normal(size=n)
    x0 = np.random.randn(d)
    x0 = x0 / np.linalg.norm(x0)

    return A, x0, b, x_star

def plot_lasso():
    A, x0, b, x_star = generate_lasso_data()

    gamma = 0.9
    sgd_t = .005
    batch_size = 32 
    iters = 2000
    epsilon = 1e-8
    B1 = .9
    B2 = .999

    l = 1.5

    sgd_mini_batch_values = sgd_mini_batch(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'lambda': l}, lasso)
    momentum_values = momentum(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma, 'lambda': l}, lasso)
    nesterov_values = nesterov(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma, 'lambda': l}, lasso)
    adagrad_values = adagrad(A, x0, b, x_star, batch_size, iters, {'rate': .1, 'epsilon': epsilon, 'lambda': l}, lasso)
    adadelta_values = adadelta(A, x0, b, x_star, batch_size, iters, {'gamma': gamma, 'epsilon': epsilon, 'lambda': l}, lasso)
    rmsprop_values = rmsprop(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma, 'epsilon': epsilon, 'lambda': l}, lasso)
    adam_values = adam(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'epsilon': epsilon, 'B1': B1, 'B2': B2, 'lambda': l}, lasso)
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


def generate_logistic_data(samples=100):
    data = datasets.load_iris()
    A = data.data[:100, :2]
    b = np.logical_not(data.target[:100])
    x_star = None
    x0 = np.zeros(2)
    return A, x0, b, x_star

def plot_logistic():
    A, x0, b, x_star = generate_logistic_data()

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
    #adagrad2_values = adagrad2(A, x0, b, x_star, batch_size, iters, {'rate': .001, 'epsilon': epsilon}, logistic)
    adadelta_values = adadelta(A, x0, b, x_star, batch_size, iters, {'gamma': gamma, 'epsilon': epsilon}, logistic)
    rmsprop_values = rmsprop(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'gamma': gamma, 'epsilon': epsilon}, logistic)
    adam_values = adam(A, x0, b, x_star, batch_size, iters, {'rate': sgd_t, 'epsilon': epsilon, 'B1': B1, 'B2': B2}, logistic)
    sgd_mini_batch_line, = plt.plot(sgd_mini_batch_values, 'g-', label='sgd mini batch error')
    momentum_line, = plt.plot(momentum_values, 'b-', label='momentum error')
    nesterov_line, = plt.plot(nesterov_values, 'y-', label='nesterov error')
    adagrad_line, = plt.plot(adagrad_values, 'k-', label='adagrad error')
    #adagrad2_line, = plt.plot(adagrad2_values, 'b-', label='adagrad error')
    adadelta_line, = plt.plot(adadelta_values, 'm-', label='adadelta error')
    rmsprop_line, = plt.plot(rmsprop_values, 'r-', label='RMSprop error')
    adam_line, = plt.plot(adam_values, 'c-', label='adam error')
    plt.legend(handles=[sgd_mini_batch_line, momentum_line, nesterov_line, adadelta_line, rmsprop_line, adam_line, adagrad_line])
    plt.show()


def main():
    hypertune_lsr()
    exit(1)
    plot_lsr()
    hypertune_logistic()
    plot_lasso()
    hypertune_lasso()
    plot_logistic()


if __name__ == '__main__':
    main()
