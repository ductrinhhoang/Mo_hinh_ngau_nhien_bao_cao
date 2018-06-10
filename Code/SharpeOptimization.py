import math
import numpy as np
import json

f = open('config.json')
json_object = json.load(f)
muy = json_object["muy"]
delta = json_object["delta"]
num_day_get = json_object["num_day_get"]

def Return(Ft, Ft1, r):
    ret = muy * (Ft * r - delta * abs(Ft - Ft1))
    return ret


def Sharpe(w, r, N, first_date):
    A = 0.0
    B = 0.0
    n = np.size(w) - 2  # n+2 size of w
    state = np.array([1.0])  # state = [1]
    for i in range(n):
        state = np.append(state, r[first_date + i])
    state = np.append(state, 0.0)
    for i in range(N - n - 1):
        F = math.tanh(w.dot(state))
        ret = Return(F, state[n + 1], r[first_date + n + i + 1])
        A += ret
        B += ret * ret
        state = np.delete(state, n + 1)
        state = np.append(state, r[first_date + n + i])
        state = np.append(state, F)
        state = np.delete(state, 0)
        state = np.delete(state, 0)
        state = np.insert(state, 0, 1.0)
    A = A/(N - n - 1)
    B = B/(N - n - 1)
    if math.sqrt(B - A * A) == 0.0:
        S = 0.0
    else:
        S = A / math.sqrt(B - A * A)
#    y = np.array([])
#    for i in range(np.size(f)):
#        y = np.append(y, i)
#    plt.gcf().clear()
    return S


def findW(w, r, N, first_date, alpha, max_loop_count, eps_for_cal_diff, eps_for_exit_loop):
    n = np.size(w) - 2
    for k in range(max_loop_count):
        dS = np.array([])
        for i in range(n + 2):
            e = np.array([0.0] * (n + 2))
            e[i] = eps_for_cal_diff
            S1 = Sharpe(w + e, r, N, first_date)
            S2 = Sharpe(w - e, r, N, first_date)
            S = (S1 - S2)/(2*eps_for_cal_diff)
            dS = np.append(dS, S)
        w1 = w + alpha * dS
        if(abs(Sharpe(w1 , r, N, first_date) - Sharpe(w , r, N, first_date)) < eps_for_exit_loop):
            w = w1
            break
        w = w1
    return w
