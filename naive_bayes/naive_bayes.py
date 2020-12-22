import numpy as np 
from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import os

def plt_fig(datas_p, labels_p, datas_n, labels_n):
    fig, ax = plt.subplots() 
    fig.set_size_inches((10, 10))
    x, y = zip(* datas_p)
    ax.scatter(x=list(x), y=list(y), color="b")
    x, y = zip(* datas_n)
    ax.scatter(x=list(x), y=list(y), color="r")
    plt.savefig("./" + "naive_bayes" + ".png", format='png', bbox_inches='tight', dpi=1000, pad_inches=0)
    plt.show()

def gen_dis(num, mean, cov, tag):
    datas = np.random.multivariate_normal(mean=mean, cov=cov, size=num).astype(np.float32)
    labels = np.repeat(tag, repeats=num).astype(np.float32)[:, np.newaxis]
    return datas, labels

def gen_data(num_p, mean_p, cov_p, tag_p, num_n, mean_n, cov_n, tag_n):
    datas_p, labels_p = gen_dis(num=num_p, mean=mean_p, cov=cov_p, tag=tag_p) 
    datas_n, labels_n = gen_dis(num=num_n, mean=mean_n, cov=cov_n, tag=tag_n)
    plt_fig(datas_p, labels_p, datas_n, labels_n)
    datas = np.concatenate((datas_p, datas_n), axis=0)
    labels = np.concatenate((labels_p, labels_n), axis=0)

    idx = list(range(len(datas)))
    # modify a sequence by shuffling its contents
    np.random.shuffle(idx)
    datas = datas[idx]
    labels = labels[idx]
    return datas, labels

class NB:
    def __init__(self):
        pass
    
    # divide data by label
    def divide_data_by_label(self, datas, labels):
        n_labels = np.squeeze(np.equal(labels, 0))
        p_labels = np.squeeze(np.equal(labels, 1))
        n_datas = datas[n_labels]
        p_datas = datas[p_labels]
        return n_datas, p_datas
        
    def calculate_para(self, datas, labels):
        n_datas, p_datas = self.divide_data_by_label(datas, labels)
        p_mu = np.mean(p_datas, axis=0)
        p_cov = np.cov(p_datas.T)
        n_mu = np.mean(n_datas, axis=0)
        p_cov = np.cov(p_datas.T)
        n_cov = np.cov(n_datas.T)

        # calculate the parameters of p(x_1 | y = positive) and p(x_2 | y = positive)
        self.p_f_1_mu = p_mu[0]
        self.p_f_1_var = p_cov[0][0]
        self.p_f_2_mu = p_mu[1]
        self.p_f_2_var = p_cov[1][1]
        
        # calculate the parameters of p(x_1 | y = negative) and p(x_2 | y = negative)
        self.n_f_1_mu = n_mu[0]
        self.n_f_1_var = n_cov[0][0]
        self.n_f_2_mu = n_mu[1]
        self.n_f_2_var = n_cov[1][1]

        # calculate the prior distribution of positive and negative classes
        self.p_y = p_datas.shape[0] / (p_datas.shape[0] + n_datas.shape[0])
        self.n_y = n_datas.shape[0] / (p_datas.shape[0] + n_datas.shape[0])

    def predict(self, x, tag_p, tag_n):
        # probability belonging to positive class
        p_prob = self.p_y * norm.pdf(x[0], self.p_f_1_mu, self.p_f_1_var) * norm.pdf(x[1], self.p_f_2_mu, self.p_f_2_var)
        # probability belonging to negative class
        n_prob = self.n_y * norm.pdf(x[0], self.n_f_1_mu, self.n_f_1_var) * norm.pdf(x[1], self.n_f_2_mu, self.n_f_2_var)
        if p_prob > n_prob:
            return tag_p
        else:
            return tag_n
        

if __name__ == "__main__":
    num_p = 1000
    mean_p = [2.5, 0.5]
    cov_p = [[0.5, 0], [0, 0.5]]
    tag_p = 1

    num_n = 2000
    mean_n = [-0.5, -0.5]
    cov_n = [[0.5, 0], [0, 0.5]]
    tag_n = 0
    
    datas, labels = gen_data(num_p, mean_p, cov_p, tag_p, num_n, mean_n, cov_n, tag_n)
    
    # calculate the parameters of naive bayes classifier
    nb = NB()
    nb.calculate_para(datas, labels)
    
    datas, labels = gen_data(100, mean_p, cov_p, tag_p, 200, mean_n, cov_n, tag_n)

    right = 0
    for data, label in zip(datas, labels):
        if nb.predict(data, tag_p, tag_n) == label:
            right += 1
    print("acc", right / len(datas))

    
