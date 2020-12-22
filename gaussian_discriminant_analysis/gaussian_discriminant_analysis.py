import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import os

def plt_fig(datas_p, labels_p, datas_n, labels_n, data_name):
    fig, ax = plt.subplots() 
    ax.set_title(data_name)
    fig.set_size_inches((10, 10))
    x, y = zip(* datas_p)
    ax.scatter(x=list(x), y=list(y), color="b")
    x, y = zip(* datas_n)
    ax.scatter(x=list(x), y=list(y), color="r")
    plt.show()

def gen_dis(num, mean, cov, tag):
    datas = list(np.random.multivariate_normal(mean=mean, cov=cov, size=num))
    labels = [tag for _ in range(num)]
    datas = np.array(datas).astype(np.float32)
    labels = np.array(labels).astype(np.float32)[:, np.newaxis]
    return datas, labels

def gen_data(num_p, mean_p, cov_p, tag_p, num_n, mean_n, cov_n, tag_n, data_name):
    datas_p, labels_p = gen_dis(num=num_p, mean=mean_p, cov=cov_p, tag=tag_p)
    datas_n, labels_n = gen_dis(num=num_n, mean=mean_n, cov=cov_n, tag=tag_n)
    plt_fig(datas_p, labels_p, datas_n, labels_n, data_name)
    datas = np.concatenate((datas_p, datas_n), axis=0)
    labels = np.concatenate((labels_p, labels_n), axis=0)
    idx = list(range(len(datas)))
    # modify a sequence by shuffling its contents
    np.random.shuffle(idx)
    datas = datas[idx]
    labels = labels[idx]
    return datas, labels

class GDA:
    def __init__(self, datas, labels):
        self.mu_1, self.mu_2, self.s, self.phi = self.get_para(datas, labels)

    def get_para(self, datas, labels):
        # datas: (num, 2)
        # labels: (num, 1)
        phi = np.sum(labels) / len(labels)

        mu_1 = np.matmul(np.transpose(labels), datas) / np.sum(labels)
        mu_2 = np.matmul(np.transpose(1 - labels), datas) / (len(labels) - np.sum(labels))
        index = np.repeat(a=labels, repeats=2, axis=-1)
        tmp_datas_1 = datas - mu_1
        # preserve required data
        tmp_datas_1 = np.multiply(index, tmp_datas_1)
        s1 = np.matmul(np.transpose(tmp_datas_1), tmp_datas_1) / np.sum(labels)

        datas_mu_2 = datas - mu_2
        datas_mu_2 = np.multiply(1 - index, datas_mu_2)
        s2 = np.matmul(np.transpose(datas_mu_2), datas_mu_2) /(len(labels) - np.sum(labels))
        s = (np.sum(labels) * s1 + (len(labels) - np.sum(labels)) * s2) / len(labels)
        mu_1 = np.squeeze(mu_1)
        mu_2 = np.squeeze(mu_2)
        return mu_1, mu_2, s, phi


    def pred(self, x):
        p1 = multivariate_normal.pdf(x, mean=self.mu_1, cov=self.s) * self.phi
        p2 = multivariate_normal.pdf(x, mean=self.mu_2, cov=self.s) * (1 - self.phi)
        if p1 > p2:
            return 1
        else:
            return 0

if __name__ == "__main__":
    ###############model solving##################
    mean_p = [1, 2]
    cov_p = [[0.1, 0.01], [0.01, 0.4]]
    num_p = 100
    tag_p = 1
    mean_n = [3, 4]
    cov_n = [[0.1, 0.01], [0.01, 0.4]]
    num_n = 50
    tag_n = 0
    datas, labels = gen_data(num_p=num_p, mean_p=mean_p, cov_p=cov_p, tag_p=tag_p,
                             num_n=num_n, mean_n=mean_n, cov_n=cov_n, tag_n=tag_n, data_name="train")

    gda = GDA(datas, labels)
    ###############model prediction##################
    num_p = 10
    num_n = 10
    datas, labels = gen_data(num_p=num_p, mean_p=mean_p, cov_p=cov_p, tag_p=tag_p,
                             num_n=num_n, mean_n=mean_n, cov_n=cov_n, tag_n=tag_n, data_name="test")
    right = 0
    all_num = len(labels)
    for i_data, i_label in zip(datas, labels):
        pred = gda.pred(i_data)
        if pred == i_label:
            right += 1
    print("accuracy: ", right / all_num)
