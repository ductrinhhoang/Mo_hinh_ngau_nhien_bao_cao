# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:12:09 2018

@author: trung dang dinh
"""

import math
import numpy as np

muy = 1000
delta = 0.01
alpha = 0.1
maxInter = 100

def Return(Ft, Ft1, r):
    ret = muy * (Ft * r - delta * abs(Ft - Ft1))
    return ret

def Sharpe(w, r, N, first_date):
    A = 0.0
    B = 0.0
    n = np.size(w) - 2 # n+2 size of w
    f = np.array([])
    state = np.array([1.0]) # state = [1]
    for i in range(n):
        state = np.append(state, r[first_date + i])
    state = np.append(state, 0.0)
    for i in range(N - n):
        F = math.tanh(w.dot(state))
        ret = Return(F, state[n + 1], r[first_date + n + i])
        A += ret
        B += ret * ret
        state = np.delete(state, n + 1)
        state = np.append(state, r[first_date + n + i])
        state = np.append(state, F)
        state = np.delete(state, 0)
        state = np.delete(state, 0)
        state = np.insert(state, 0, 1.0)
    A = A/(N - n)
    B = B/(N - n)
    if math.sqrt(B - A * A) == 0.0:
        S = 0.0
    else:
        S = A / math.sqrt(B - A * A)
    y = np.array([])
    for i in range(np.size(f)):
        y = np.append(y, i)
#    plt.gcf().clear()
    return S

def findW(w, r, N, first_date):
    n = np.size(w) - 2
    eps = 0.1
    for k in range(maxInter):
        e = np.array([0.0] * (n + 2))
        dS = np.array([])
        for i in range(n + 2):
            e[i] = eps
            S1 = Sharpe(w + e, r, N, first_date)
            S2 = Sharpe(w - e, r, N, first_date)
            S = (S1 - S2)/(2*eps)
            dS = np.append(dS, S)
        w = w + alpha * dS
    return w