# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 15:12:09 2018

@author: trung dang dinh
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import xlrd

#Get data from excel file
file_location = "C:/Users/trung dang dinh/Desktop/reinforcement learning/daily_MSFT_full.xlsx"
wb = xlrd.open_workbook(file_location)
sheet = wb.sheet_by_index(0)

price = []
#i = 1
#while i <= sheet.nrows - 1:
i = 1
while i < 4000:
    price.append(sheet.cell_value(i, 4))
    i += 1

y = np.array([])
for i in range(np.size(price)):
    y = np.append(y, i)

#plt.plot(y, price)
#plt.show

N = 255

#rt
r = np.array([])
for i in range(N):
    r = np.append(r, price[i + 1] - price[i])

first_date = 5
n = 5

w = np.array([0] * (n + 2))
F = 0
muy = 1000
delta = 0.01
alpha = 0.1
maxInter = 100

def Return(Ft, Ft1, r):
    ret = muy * (Ft * r - delta * abs(Ft - Ft1))
    return ret

def Sharpe(w):
    A = 0.0
    B = 0.0
    state = np.array([1.0])
    for i in range(n):
        state = np.append(state, r[i])
    state = np.append(state, 0.0)
    for i in range(N - n):
        F = math.tanh(w.dot(state))
        ret = Return(F, state[n + 1], r[n + i])
        A += ret
        B += ret * ret
        state = np.delete(state, n + 1)
        state = np.append(state, r[n + i])
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
    return S

def findW(w):
    eps = 0.1
    for k in range(maxInter):
        e = np.array([0.0] * (n + 2))
        dS = np.array([])
        for i in range(n + 2):
            e[i] = eps
            S1 = Sharpe(w + e)
            S2 = Sharpe(w - e)
            S = (S1 - S2)/(2*eps)
            dS = np.append(dS, S)
        w = w + alpha * dS
    return w

print(Sharpe(findW(w)))