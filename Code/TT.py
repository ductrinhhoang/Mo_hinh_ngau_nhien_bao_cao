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
    num_day_get = json_object["num_day_get"]
    eps_for_cal_diff = json_object["eps_for_cal_diff"]
    eps_for_exit_loop = json_object["eps_for_exit_loop"]
    # paremeter

    # Get data from excel file
    wb = xlrd.open_workbook(file_location)
    sheet = wb.sheet_by_index(0)

    price = get_price(sheet, num_day_get)

    size = np.size(price)
    # y = np.array([])
    # for i in range(size):
    #     y = np.append(y, i)
    # plt.plot(y, price)
    # plt.show()

    r = price[1:size] - price[0:size-1]

    w = np.array([0] * (n + 2))

    # print(SO.Sharpe(SO.findW(w, r, N, 0), r, N, 0))
    RET = np.array([])
    F = 0.0  # Ft time -1
    count = N + 1
    while count < (size - 1):
        if (count - N) % m == 1:
            w = SO.findW(w, r, N, count - N - 1, alpha,
                         max_loop_count, eps_for_cal_diff, eps_for_exit_loop)
        state = np.array([1.0])  # state = [1]
        for i in range(n):
            state = np.append(state, r[count - i])
        state = np.append(state, F)
        Ft = math.tanh(w.dot(state))
        ret = SO.Return(F, Ft, r[count])
        RET = np.append(RET, ret)
        F = Ft
        count += 1

    gain = np.cumsum(RET)
    print(gain)
    time_series = np.array(range(np.size(gain)))

    print("Thời gian chạy: ", TIME.time() - start_time)
    plt.plot(time_series, gain)
    plt.show()


if __name__ == "__main__":
    main()
