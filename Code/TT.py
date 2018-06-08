# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 19:04:05 2018

@author: trung dang dinh
"""

import SharpeOptimization as SO
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
while i < 1000:
    price.append(sheet.cell_value(i, 4))
    i += 1

size = np.size(price)
y = np.array([])
for i in range(size):
    y = np.append(y, i)

#plt.plot(y, price)
#plt.show()

n = 5 #number of days in state
N = 255 #number of days for training
m = 5 #number of days for trading

r = np.array([])
for i in range(size - 1):
    r = np.append(r, price[i + 1] - price[i])

first_date = 5

w = np.array([0] * (n + 2))
    
# print(SO.Sharpe(SO.findW(w, r, N, 0), r, N, 0))
RET = np.array([])
F = 0.0 #Ft time -1
count = N + 1
while count < (size - 1):
    if (count - N) % m == 1:
        w = SO.findW(w, r, N, count - N - 1)
    state = np.array([1.0]) # state = [1]
    for i in range(n):
        state = np.append(state, r[count - i])
    state = np.append(state, F)
    Ft = math.tanh(w.dot(state))
    ret = SO.Return(F, Ft, r[count])
    RET = np.append(RET, ret)
    F = Ft
    count += 1

gain = np.cumsum(RET)
time = np.array([])
for i in range(np.size(gain)):
    time = np.append(time, i)

plt.plot(time, gain)
plt.show()