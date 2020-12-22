#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def gen_dis(num, mean, cov, tag):
    datas = list(np.random.multivariate_normal(mean=mean, cov=cov, size=num))
    labels = [tag for _ in range(num)]
    datas = np.array(datas)
    labels = np.array(labels)
    return datas, np.array(labels, dtype=np.float32)

def plt_fig(p_datas, p_labels, n_datas, n_labels):
    fig, ax = plt.subplots()
    fig.set_size_inches((10, 10))
    x, y = zip(* p_datas)
    ax.scatter(x=list(x), y=list(y), color="b")
    x, y = zip(* n_datas)
    ax.scatter(x=list(x), y=list(y), color="r")
    plt.show()

def gen_data(pos_num, pos_mean, pos_cov, neg_num, neg_mean, neg_cov):
    p_datas, p_labels = gen_dis(pos_num, pos_mean, pos_cov, tag=1)
    n_datas, n_labels = gen_dis(neg_num, neg_mean, neg_cov, tag=-1)
    #plt_fig(p_datas, p_labels, n_datas, n_labels)
    all_datas = np.concatenate((p_datas, n_datas), axis=0)
    all_labels = np.concatenate((p_labels, n_labels), axis=0)
    indexes = list(range(len(all_labels)))
    np.random.shuffle(indexes)
    all_datas = all_datas[indexes]
    all_labels = all_labels[indexes]
    return all_datas, all_labels

def lda(datas, labels):
    p_samples_x = list()
    p_samples_y = list()
    n_samples_x = list()
    n_samples_y = list()
    for x, y in zip(datas, labels):
        if y > 0 :
            p_samples_x.append(x)
            p_samples_y.append(y)
        else:
            n_samples_x.append(x)
            n_samples_y.append(y)
    
    p_avg_x = np.mean(p_samples_x, 0)
    n_avg_x = np.mean(n_samples_x, 0)
    p_samples_x = p_samples_x - p_avg_x
    n_samples_x = n_samples_x - n_avg_x
    p_var_x = np.matmul(np.transpose(p_samples_x), p_samples_x) / (p_samples_x.shape[0])
    n_var_x = np.matmul(np.transpose(n_samples_x), n_samples_x) / (n_samples_x.shape[0])
    tmp_diff = (p_avg_x - n_avg_x) 
    tmp_diff = tmp_diff[:, np.newaxis]
    tmp_diff_T = np.transpose(tmp_diff)
    s_b = np.matmul(tmp_diff, tmp_diff_T)
    s_w = np.add(p_var_x, n_var_x).astype(np.float32)
    avg_diff = p_avg_x - n_avg_x
    avg_diff = avg_diff[:, np.newaxis].astype(np.float32)
    w = np.dot(np.linalg.inv(s_w), avg_diff)
    return w

def draw_picture(datas, labels, w_form):
    p_samples = [it for it, label in zip(datas, labels) if label > 0]
    x, y = zip(* p_samples)
    p_samples_x = list(x)
    p_samples_y = list(y)

    n_samples = [it for it, label in zip(datas, labels) if label < 0]
    x, y = zip(* n_samples)
    n_samples_x = list(x)
    n_samples_y = list(y)
    fig, ax = plt.subplots()
    ax.scatter(p_samples_x, p_samples_y, c="red")
    ax.scatter(n_samples_x, n_samples_y, c="blue")

    w1 = w_form[0][0]
    w2 = w_form[1][0]
    x_list = np.linspace(min(p_samples_x + n_samples_x), max(p_samples_x + n_samples_x), num=100)
    y_list = list()
    for x in x_list:
        y_list.append(x * w2 / w1)
    ax.plot(x_list, y_list, "g--")
    fig.set_size_inches((10, 10))
    plt.savefig("lda_result.png", format="png", dpi=1000, bbox_inches="tight", pad_inches=0)
    plt.show()

if __name__ == "__main__":
    # 生成数据
    p_num = 200
    p_mean = np.array([2, 2])
    p_cov = np.array([[0.5, 0.2], [0.2, 0.6]])
    n_num = 300
    n_mean = np.array([4, 4])
    n_cov = np.array([[0.3, 0.1], [0.1, 0.2]])
    
    datas, labels = gen_data(p_num, p_mean, p_cov, n_num, n_mean, n_cov)
    w = lda(datas, labels)
    draw_picture(datas, labels, w)
