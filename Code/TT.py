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
import json
import time as TIME


def get_price(sheet, num_day_get):
    price = []
    for i in range(1, num_day_get):
        price.append(sheet.cell_value(i, 4))
    return np.array(price)


def get_sharpe(RET):
    sharpe = np.array([0])
    for i in range(1, len(RET)):
        A = sum(RET[1:1+i])/(i+1)
        B = RET[1:i+1].dot(RET[1:i+1])/(i+1)
        if math.sqrt(B - A**2) == 0.0:
            sharpe = 0.0
        else:
            sharpe = np.append(sharpe, math.sqrt(i+1)*A/(math.sqrt(B-A**2)))
    return sharpe


def main():
    start_time = TIME.time()
    # read json
    f = open('config.json')
    json_object = json.load(f)

    file_location = json_object['open_path']
    alpha = json_object["alpha"]
    max_loop_count = json_object["max_loop_count"]
    n = json_object["num_day_in_state"]  # number of days in state
    N = json_object["num_day_for_train"]  # number of days for training
    m = json_object["num_day_for_trade"]  # number of days for trading
    T = json_object["num_day_get"]
    eps_for_cal_diff = json_object["eps_for_cal_diff"]
    eps_for_exit_loop = json_object["eps_for_exit_loop"]
    # paremeter

    # Get data from excel file
    wb = xlrd.open_workbook(file_location)
    sheet = wb.sheet_by_index(0)

    price = get_price(sheet, T+1)
#    y = np.array(range(np.size(price)))
#    plt.plot(y, price)
#    plt.show()

    r = price[1:T] - price[0:T-1]

    w = np.array([0] * (n + 2))

    RET = np.array([0])
    F = 0.0  # Ft time -1
    F_array = np.array([])
    count = N + 1
    while count < T:
        if (count - N) % m == 1 or m == 1:
            w = SO.findW(w, r, N, count - N - 1, alpha,
                         max_loop_count, eps_for_cal_diff, eps_for_exit_loop)
        state = np.array([1.0])  # state = [1]
        for i in range(n):
            state = np.append(state, r[count - i - 2])
        state = np.append(state, F)
        Ft = math.tanh(w.dot(state))
        F_array = np.append(F_array, Ft)
        # print(len(F_array), count - 1)
        if count < T - 1: # caculate ret at tomorrow
            ret = SO.Return(Ft, F, r[count])
            RET = np.append(RET, ret)
        F = Ft
        count += 1

    gain = np.cumsum(RET)
    time_series = np.array(range(np.size(gain)))

    print("Thời gian chạy: ", TIME.time() - start_time)

    plt.subplot(3, 1, 1)
    plt.title("PnL")
    plt.plot(time_series, gain)

    plt.subplot(3, 1, 2)
    time_series = np.array(range(np.size(F_array)))
    plt.title("Long-Short")
    plt.plot(time_series, F_array)

    plt.subplot(3, 1, 3)
    plt.title("Sharpe")
    real_sharpe = get_sharpe(RET)
    time_series = np.array(range(np.size(real_sharpe)))
    plt.plot(time_series, real_sharpe)

    plt.show()


if __name__ == "__main__":
    main()
