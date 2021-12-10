import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize


def functions(x, t, num):
    t = (t+1)/10 + t % 10
   # t = t//10 + 1
    if num > 5:
        num = int(num - 6)
        fn = gen_fun(num)
        res = fn(t * x)
        aux = t*x
    else:
        fn = gen_fun(num)
        res = fn(t + x)
        aux = t+x

    res = res + np.random.randn(res.shape[0])
    aux = aux+ np.random.randn(aux.shape[0])
    return (res, aux)


def fn1(x):
    return x ** 0.8

def fn2(x):
    return x ** 1.2

def gen_fun(i):
    a = np.sin
    b = np.cos
    c = np.sqrt
    d = np.log
    e = fn1
    f = fn2
    ls = [a, b, c, d, e, f]
    return ls[i]


# Here we generate a list of probabilities for each group
# e denotes number of donors
def gen_Probs(e, k, s):
    # e is the number of donors
    # the total is e + 1
    e = e + 1
    Probs = []
    for i in range(e):
        # constructing the probability distribution for the target
        if i == 0:
            alphas = np.ones(k)
            p = np.random.dirichlet(alphas, size=1)[0]
        else:
            # constructing the probability distribution for the target
            p = 1 - (s / k)
            index = np.random.binomial(size=k, n=1, p=p)
            if np.sum(index) == 0:
                zero = True
                while zero:
                    index = np.random.binomial(size=k, n=1, p=p)
                    zero = np.sum(index) == 0
            alphas = index + 0.001
            p = np.random.dirichlet(alphas, size=1)[0]
            if np.isnan(p)[0]:
                nan = True
                while nan:
                    p = np.random.dirichlet(alphas, size=1)[0]
                    nan = np.isnan(p)[0]
            p = p * index / (np.sum(p * index))
        Probs.append(p)
    return Probs


def make_matrix(Probs, k, T, n, summary='aggregate'):
    tmp = abs(np.random.randn(T)).reshape(-1, 1)
    matrix = np.zeros((len(Probs), T))
    for state in range(len(Probs)):
        idx_state = np.random.choice(np.arange(1, k + 1), size=n, p=Probs[state])
        for t in range(T):
            t_pop = []
            tmps = []
            for i in range(k):
                index = idx_state == i + 1
                if np.sum(index) > 0:
                    c_jt = idx_state[index] * tmp[t]
                    inds, aux = functions(c_jt, t, i)
                    t_pop.append(inds)
                    tmps.append(aux)
            pop = np.concatenate(t_pop)
            # print("the size of population", len(pop))
            if summary == 'aggregate':
                matrix[state][t] = np.mean(pop)
            elif summary == 'median':
                matrix[state][t] = np.quantile(pop, 0.5)
            elif summary == 'aux_good':
                aux = np.sin(pop) + np.random.randn(pop.shape[0])
                matrix[state][t] = np.mean(aux)
            elif summary == 'aux_bad':
                concat = np.concatenate(tmps)
                matrix[state][t] = np.mean(concat)/10
    return matrix


def fit_regression(matrix, T, convex=False, aux=False, train=0):
    Donors = np.array(matrix[1:]).T
    Target = np.array(matrix[0])
    if aux:
        train = train
    else:
        train = int(3 * T / 4)
    train_X = Donors[:train]
    test_X = Donors[train:]
    train_Y = Target[:train]
    test_Y = Target[train:]

    if convex:
        reg = ConvexRegression(train_X, train_Y)
        train_pred = train_X @ reg.x
        test_pred = test_X @ reg.x
    else:
        reg = LinearRegression(fit_intercept=False)
        reg.fit(train_X, train_Y)
        train_pred = reg.predict(train_X)
        test_pred = reg.predict(test_X)

    train_loss = _mse(train_pred, train_Y)
    test_loss = _mse(test_pred, test_Y)
    return (train_loss, test_loss)


def _mse(pred, y):
    pred = pred.reshape(-1, 1)
    y = y.reshape(-1, 1)
    return np.mean(np.square(pred - y))


def ConvexRegression(X, y):
    p, n = X.shape

    # Objective function
    def f(w):
        return ((np.dot(X, w) - y) ** 2).sum()

    def jac_f(w):
        return (-(2 * ((y - np.dot(X, w)).T).dot(X)))

    # Defining constraints
    def sum_con(w):
        return (np.ones((n)).dot(w) - 1)

    dic_sum_con = {"type": "eq", "fun": sum_con}

    def positive_con(w):
        return w

    dic_positive_con = {"type": "ineq", "fun": positive_con}

    cons = [dic_sum_con, dic_positive_con]

    # Scipy optimization
    result = minimize(f, np.ones(n) / n, jac=jac_f, constraints=cons, method="SLSQP")

    return result
