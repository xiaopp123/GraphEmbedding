# -*- coding: utf-8 -*-

"""
Alias采样算法
https://www.cnblogs.com/Lee-yl/p/12749070.html
"""


import numpy as np


def create_alias_table(area_ratio):
    """
     area_ratio[i]代表事件i出现的概率
     :param area_ratio: sum(area_ratio)=1
     :return: accept,alias
    """
    N = len(area_ratio)
    accept, alias = [0] * N, [0] * N
    small, large = [], []
    # ------（1）概率 * N -----
    area_ratio_ = np.array(area_ratio) * N
    # ------（2）获取small 、large -----
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)
    # ------（3）修改柱状图 ----- （4）获取accept和alias -----
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        area_ratio_[large_idx] = area_ratio_[large_idx] - \
                                 (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    while large:
        large_idx = large.pop()
        accept[large_idx] = 1

    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """随机采样1~N之间的整数i，决定落在哪一列。
    随机采样0~1之间的一个概率值，
    如果小于accept[i]，则采样i，
    如果大于accept[i]，则采样alias[i]；
    :param accept:
    :param alias:
    :return:
    """
    N = len(accept)
    i = int(np.random.random()*N)
    r = np.random.random()
    if r < accept[i]:
        return i
    else:
        return alias[i]